"""Check final-training admission and restart metadata without numerical imports."""

from collections import namedtuple
import copy
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train_optimized as training


HostState = namedtuple("HostState", sorted(training.STATE_FIELDS))


def example_specification():
    """Supply a small metadata fixture with the full fixed final-training dimensions."""
    return {"schema_version": 1, "source_root": "/immutable/source", "source_commit": "a" * 40,
            "hardware_class": "h100", "copy_seeds": [1, 2], "final_verification": "/evidence/final.json",
            "asset_record": {"root": "/assets", "manifest": "/evidence/assets.json", "sha256": "c" * 64},
            "runtime_environment": {"JAX_ENABLE_X64": "true", "JAX_DEFAULT_MATMUL_PRECISION": "highest",
                                    "XLA_FLAGS": "--xla_gpu_exclude_nondeterministic_ops=true"},
            "configuration": {"learner": {"n_copies": 2, "total_timesteps": 2_000_000_000,
                "n_envs": 128, "num_steps": 128, "update_epochs": 4, "num_minibatches": 4},
                "environment": {"num_envs": 128}, "execution": {"compilation_mode": "whole_iteration",
                    "iterations_per_call": 1, "report_readout": "none", "report_every_iterations": 10}},
            "validation": {"kind": "training_admission", "raw_evidence": "fixture only"}}


def example_identity():
    """Bind the fixture to deterministic settings and explicit helper/backend receipts."""
    specification = example_specification()
    return training.checkpoint_identity(specification, {"commit": "b" * 40},
                                        specification["runtime_environment"], {"jax": "0.11.0"}, {"tree_digest": "assets"})


def example_state(iteration=3):
    """Keep all named state components present so serialization cannot silently select a subset."""
    values = {name: {"preserved_field": name} for name in training.STATE_FIELDS}
    values.update(iteration=iteration, copy_seeds=[1, 2])
    return HostState(**values)


class TrainingMetadataTests(unittest.TestCase):
    """Reject altered identities and preserve complete state and execution boundaries."""

    def test_fixed_contract_and_no_hyperparameter_search(self):
        """Changing a batch, budget, seed assignment, or reporting boundary fails early."""
        self.assertEqual(training.validate_specification(example_specification())["copy_seeds"], [1, 2])
        for name, value in (("total_timesteps", 1_000_000), ("n_envs", True), ("num_steps", 64),
                            ("update_epochs", 1), ("num_minibatches", 2)):
            specification = example_specification()
            specification["configuration"]["learner"][name] = value
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "learner contract"):
                training.validate_specification(specification)
        for seeds in ([1, 1], [1], [1, 2 ** 32], [True, 2]):
            specification = example_specification()
            specification["copy_seeds"] = seeds
            with self.subTest(seeds=seeds), self.assertRaisesRegex(ValueError, "copy_seeds"):
                training.validate_specification(specification)
        specification = example_specification()
        specification["configuration"]["learner"]["per_copy_learning_rates"] = [1e-4, 1e-3]
        with self.assertRaisesRegex(ValueError, "hyperparameters"):
            training.validate_specification(specification)
        specification = example_specification()
        specification["configuration"]["execution"].update(iterations_per_call=5, report_readout="scheduled")
        with self.assertRaisesRegex(ValueError, "boundaries"):
            training.validate_specification(specification)

    def test_component_commit_cannot_ascend_to_a_parent_repository(self):
        """Use an explicit archive identity or an exact Git root, never a moving ancestor HEAD."""
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            archive = parent / ("a" * 40)
            archive.mkdir()
            with mock.patch.object(training.subprocess, "check_output") as git:
                self.assertEqual(training.component_source_commit(archive, {
                    "RND_COMPONENT_SOURCE_COMMIT": "a" * 40}), "a" * 40)
                git.assert_not_called()
            for supplied in ("a" * 12, "b" * 40, "g" * 40):
                with self.subTest(supplied=supplied), self.assertRaisesRegex(ValueError, "archive"):
                    training.component_source_commit(archive, {"RND_COMPONENT_SOURCE_COMMIT": supplied})
            with mock.patch.object(training.subprocess, "check_output", return_value=str(parent)) as git:
                with self.assertRaisesRegex(ValueError, "ancestor"):
                    training.component_source_commit(archive, {})
                self.assertEqual(git.call_count, 1)
            with mock.patch.object(training.subprocess, "check_output", side_effect=[str(archive), "c" * 40]):
                self.assertEqual(training.component_source_commit(archive, {}), "c" * 40)

    def test_boolean_only_admission_fails_before_source_or_numerical_import(self):
        """A supplied permission flag cannot stand in for the validator's evidence API."""
        validator = types.ModuleType("post_comparison_validation")
        validator.validate_training_admission = mock.Mock(return_value={"authorized": True})
        before = set(sys.modules)
        with mock.patch.dict(sys.modules, {"post_comparison_validation": validator}), \
                mock.patch.object(training, "source_receipt") as receipt:
            with self.assertRaisesRegex(ValueError, "schema_version"):
                training.run(types.SimpleNamespace(final_spec=Path("/unverified.json")))
            receipt.assert_not_called()
        self.assertFalse({"jax", "numpy", "torch"} & (set(sys.modules) - before))

    def test_runtime_matches_validated_flags_and_records_extended_environment(self):
        """Equivalent whitespace is accepted; conflicting precision or compiler flags is rejected."""
        required = example_specification()["runtime_environment"]
        environment = {"XLA_FLAGS": "  --xla_gpu_exclude_nondeterministic_ops=true  ",
                       "CUDA_MODULE_LOADING": "LAZY", "CUDA_VISIBLE_DEVICES": "1"}
        actual = training.enforce_runtime(required, environment)
        self.assertEqual(actual["JAX_ENABLE_X64"], "true")
        self.assertEqual(actual["CUDA_MODULE_LOADING"], "LAZY")
        self.assertEqual(actual["JAX_ENABLE_COMPILATION_CACHE"], "false")
        self.assertEqual(actual["JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES"], "")
        self.assertNotIn("CUDA_VISIBLE_DEVICES", actual)
        for environment in ({"JAX_ENABLE_X64": "false"}, {"JAX_DEFAULT_MATMUL_PRECISION": "tensorfloat32"},
                            {"XLA_FLAGS": "--xla_gpu_exclude_nondeterministic_ops=false"}):
            with self.subTest(environment=environment), self.assertRaisesRegex(ValueError, "differs"):
                training.enforce_runtime(required, environment)
        with self.assertRaisesRegex(ValueError, "exactly once"):
            training.enforce_runtime({**required, "XLA_FLAGS": required["XLA_FLAGS"] + " " + required["XLA_FLAGS"]}, {})
        for environment in ({"JAX_COMPILATION_CACHE_DIR": "/unsafe/cache"},
                            {"JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES": "all"},
                            {"JAX_ENABLE_COMPILATION_CACHE": "true"}):
            with self.subTest(environment=environment), self.assertRaisesRegex(ValueError, "cache"):
                training.enforce_runtime(required, environment)

    def test_checkpoint_retains_every_state_component_and_rejects_changed_identity(self):
        """Round-trip the full state and reject altered source, settings, seeds, flags, or backend."""
        identity = example_identity()
        report = {"pending_sums": None, "pending_count": 0, "last_report_iteration": 0}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "checkpoints/latest.pkl"
            state = example_state()
            training.save_checkpoint(path, state, report, identity, 3, 12.5)
            loaded = training.load_checkpoint(path, identity)
            self.assertEqual(loaded["state"], state)
            self.assertEqual(set(loaded["state"]._fields), training.STATE_FIELDS)
            self.assertEqual(loaded["elapsed_seconds"], 12.5)
            checkpoint_metadata = json.loads(path.with_name("latest.metadata.json").read_text())
            self.assertEqual(checkpoint_metadata["iteration"], 3)
            self.assertEqual(checkpoint_metadata["copy_seeds"], [1, 2])
            self.assertEqual(checkpoint_metadata["checkpoint_sha256"], training.file_sha256(path))
            for name in ("source_commit", "configuration", "copy_seeds", "runtime_environment", "backend", "helpers", "assets"):
                changed = copy.deepcopy(identity)
                changed[name] = {"different": "identity"}
                with self.subTest(name=name), self.assertRaisesRegex(ValueError, "identity differs"):
                    training.load_checkpoint(path, changed)
            with self.assertRaisesRegex(ValueError, "actual learner state"):
                training.save_checkpoint(path, state, report, identity, 4, 12.5)
            # The rejected replacement must leave the prior valid checkpoint intact.
            self.assertEqual(training.load_checkpoint(path, identity)["iteration"], 3)

    def test_pending_report_checkpoint_keeps_chronological_position(self):
        """A partial window resumes at its true update rather than restarting the interval."""
        identity = example_identity()
        identity["configuration"]["execution"]["report_readout"] = "scheduled"
        report = {"pending_sums": {"preserved": "metrics"}, "pending_count": 3, "last_report_iteration": 0}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "latest.pkl"
            training.save_checkpoint(path, example_state(), report, identity, 3, 1)
            self.assertEqual(training.load_checkpoint(path, identity)["report_state"], report)
        for changed in ({**report, "pending_count": 2}, {**report, "last_report_iteration": 1},
                        {**report, "pending_sums": None}):
            with self.subTest(changed=changed), self.assertRaisesRegex(ValueError, "window"):
                training.validate_report_state(changed, identity["configuration"]["execution"], 3)

    def test_group_tail_and_schedule_boundary_do_not_pad_updates(self):
        """For example, seven remaining updates become five plus one plus one."""
        calls, iteration = [], 122063
        while iteration < 122070:
            count = training.next_update_count(iteration, 122070, 5)
            calls.append(count)
            iteration += count
        self.assertEqual(calls, [5, 1, 1])
        self.assertEqual(training.next_update_count(122070, 122070, 5), 0)
        with self.assertRaises(ValueError):
            training.next_update_count(122071, 122070, 5)

    def test_execution_dispatch_uses_group_and_original_tail(self):
        """Dispatch full groups through Executor and only short tails through its one-update call."""
        executor = types.SimpleNamespace(group_size=5, options={"report_readout": "none"},
            run=mock.Mock(return_value=("group state", "group metrics")),
            one=mock.Mock(return_value=("tail state", "tail metrics")))
        execution = training.TrainingExecution(executor, None, None, None, None)
        self.assertEqual(execution.advance("state", 5, 10), ("group state", "group metrics", []))
        executor.run.assert_called_once_with("state", 5, initial_iteration=10)
        self.assertEqual(execution.advance("state", 1, 15), ("tail state", "tail metrics", []))
        executor.one.assert_called_once_with("state")
        with self.assertRaisesRegex(ValueError, "group or tail"):
            execution.advance("state", 2, 16)

    def test_scheduled_window_continues_after_host_checkpoint(self):
        """The first report after resume finishes the existing window without an extra host read per update."""
        options = {"report_readout": "scheduled", "report_every_iterations": 3}
        executor = types.SimpleNamespace(group_size=1, options=options,
                                        one=mock.Mock(return_value=("next state", "new metric")))
        paired = types.SimpleNamespace(accumulate_report=mock.Mock(return_value="completed device sums"),
                                       finish_report=mock.Mock(return_value={"report": "complete"}))
        jax = types.SimpleNamespace(device_get=mock.Mock(return_value="host sums"))
        pending = {"pending_sums": "resumed sums", "pending_count": 2, "last_report_iteration": 0}
        execution = training.TrainingExecution(executor, paired, jax, "host library", "device library", pending)
        state, metrics, reports = execution.advance("restored state", 1, 2)
        paired.accumulate_report.assert_called_once_with("resumed sums", "new metric", "device library")
        jax.device_get.assert_called_once_with("completed device sums")
        self.assertEqual(reports, [{"iteration": 3, "window_updates": 3, "metrics": {"report": "complete"}}])
        self.assertEqual(execution.report, {"pending_sums": None, "pending_count": 0, "last_report_iteration": 3})
        self.assertEqual((state, metrics), ("next state", "new metric"))

    def test_resume_attempts_preserve_previous_failures(self):
        """A new process attempt has its own directory while the old failure remains readable."""
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            number, first = training.new_attempt(directory)
            self.assertEqual(number, 1)
            (first / "failure.json").write_text(json.dumps({"error": "earlier process"}))
            number, second = training.new_attempt(directory)
            self.assertEqual(number, 2)
            self.assertTrue((first / "failure.json").exists())
            self.assertFalse((second / "failure.json").exists())

    def test_existing_outputs_cannot_be_reused_for_another_specification(self):
        """A final-unit name cannot overwrite a benchmark or adopt a different seed assignment."""
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            original = directory / "progress.json"
            original.write_text("original benchmark record")
            with self.assertRaisesRegex(ValueError, "non-training experiment"):
                training.prepare_unit(directory, example_specification())
            self.assertEqual(original.read_text(), "original benchmark record")
            original.unlink()
            training.prepare_unit(directory, example_specification())
            training.prepare_unit(directory, example_specification())
            changed = example_specification()
            changed["copy_seeds"] = [3, 4]
            with self.assertRaisesRegex(ValueError, "different specification"):
                training.prepare_unit(directory, changed)

    def test_runtime_source_must_match_committed_helper_bytes(self):
        """Changed helper contents fail even when the claimed commit has not changed."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "helper.py").write_bytes(b"committed source\n")
            replies = ["a" * 40 + "\n", "", b"committed source\n"]
            with mock.patch.object(training.subprocess, "check_output", side_effect=replies):
                receipt = training.source_receipt(root, "a" * 40, ["helper.py"])
            self.assertIn("helper.py", receipt["files"])
            with mock.patch.object(training.subprocess, "check_output", side_effect=["b" * 40 + "\n"]):
                with self.assertRaisesRegex(ValueError, "exact admitted commit"):
                    training.source_receipt(root, "a" * 40, ["helper.py"])
            with mock.patch.object(training.subprocess, "check_output", side_effect=["a" * 40 + "\n", "", b"older\n"]):
                with self.assertRaisesRegex(ValueError, "Git snapshot"):
                    training.source_receipt(root, "a" * 40, ["helper.py"])

    def test_asset_manifest_rejects_changed_and_unrecorded_inputs(self):
        """Verify the full input tree instead of trusting an XDG directory name."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "assets"
            root.mkdir()
            sprite = root / "sprite.png"
            sprite.write_bytes(b"recorded sprite bytes")
            manifest = Path(temporary) / "asset_manifest.json"
            manifest.write_text(json.dumps({"schema_version": 1, "source_commit": training.JAXTARI_COMMIT,
                                            "files": {"sprite.png": training.file_sha256(sprite)}}))
            record = {"root": str(root), "manifest": str(manifest), "sha256": training.file_sha256(manifest)}
            environment = {}
            receipt = training.verify_assets(record, environment)
            self.assertEqual(receipt["file_count"], 1)
            self.assertEqual(environment["XDG_DATA_HOME"], str(root))
            sprite.write_bytes(b"changed sprite bytes")
            with self.assertRaisesRegex(ValueError, "differs"):
                training.verify_assets(record, {})
            sprite.write_bytes(b"recorded sprite bytes")
            (root / "unrecorded.png").write_bytes(b"extra")
            with self.assertRaisesRegex(ValueError, "exactly"):
                training.verify_assets(record, {})


if __name__ == "__main__":
    unittest.main()
