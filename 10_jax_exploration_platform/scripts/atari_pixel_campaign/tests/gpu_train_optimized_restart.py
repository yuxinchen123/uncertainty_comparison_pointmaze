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
import os
from pathlib import Path
import socket
import sys
import tempfile
import unittest


if os.environ.get("RND_RUN_GPU_COMPONENT_TESTS") != "1":
    raise unittest.SkipTest("GPU component checks require an explicit compute-only invocation")
if not os.environ.get("SLURM_JOB_ID") and socket.gethostname().split(".")[0] not in {"serval02", "jaguar03"}:
    raise RuntimeError("GPU component checks cannot run on a login host")

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SCRIPT_ROOT.parents[2]
sys.path.insert(0, str(SCRIPT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "10_jax_exploration_platform/src"))
import train_optimized as training

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
        """A saved three-update group resumes through one more group and one original tail update."""
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
            # Independent leaf buffers also exercise restart with donation enabled.
            state = paired.device_copy(restored["state"])
            execution = self.execution(options, restored["report_state"], self.restarted_trainer())
            calls = []
            iteration = int(jax.device_get(state.iteration))
            while iteration < self.config.total_iterations:
                count = training.next_update_count(iteration, self.config.total_iterations, execution.executor.group_size)
                state, metrics, _ = execution.advance(state, count, iteration)
                jax.block_until_ready(state)
                actual = int(jax.device_get(state.iteration))
                self.assertEqual(actual, iteration + count)
                calls.append(count)
                iteration = actual
            self.assertEqual(calls, [3, 1])
            training.save_checkpoint(checkpoint, jax.device_get(state), jax.device_get(execution.report),
                                     identity, iteration, 2.0)
            self.assertEqual(training.load_checkpoint(checkpoint, identity)["iteration"], 7)
        self.assert_state_agreement(uninterrupted, state, execution.executor)
        paired.agreement(jax.device_get(reference_metrics), jax.device_get(metrics))

    def test_scheduled_pending_report_resume_matches_uninterrupted_executor(self):
        """Save two updates into a four-update report window and preserve its next published row."""
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
            restored = training.load_checkpoint(checkpoint, identity)
            self.assertEqual(restored["report_state"]["pending_count"], 2)
            restored["report_state"]["pending_sums"] = paired.device_copy(restored["report_state"]["pending_sums"])
            resumed = self.execution(options, restored["report_state"], self.restarted_trainer())
            state = paired.device_copy(restored["state"])
            reports = []
            for iteration in range(2, 7):
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
