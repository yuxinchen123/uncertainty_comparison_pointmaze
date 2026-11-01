"""Check adaptive environment metadata without importing numerical packages or starting jobs."""

import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_ROOT))
import controller
import ideas
import run_case


class ReferenceEnvironmentTests(unittest.TestCase):
    """Keep the agreed game protocol fixed while carrying accepted execution settings."""

    def setUp(self):
        """Prepare a temporary source admission with explicit synthetic check records."""
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.run = Path(self.temporary.name) / "campaign"
        self.run.mkdir()
        self.source = SCRIPT_ROOT.parents[2]
        self.commit = "a" * 40
        self.configuration = self.run / "reference_configuration.json"
        controller.atomic_json(self.configuration, controller.REFERENCE_ALGORITHM)
        gates = {}
        for context in ideas.REQUIRED_HARDWARE:
            path = self.run / f"{context}_check.json"
            controller.atomic_json(path, {"passed": True, "source_commit": self.commit})
            gates[context] = str(path)
        self.sentinel = self.run / controller.SENTINEL_NAME
        self.ready = {"authorized": True, "source_root": str(self.source),
                      "source_commit": self.commit, "contexts": list(ideas.REQUIRED_HARDWARE),
                      "matched_copy_counts": [1], "gate_records": gates,
                      "reference_configuration_path": str(self.configuration)}
        controller.atomic_json(self.sentinel, self.ready)
        for module in (controller, run_case):
            patch = mock.patch.object(module, "verify_source", return_value={
                "source_root": str(self.source), "source_commit": self.commit})
            patch.start()
            self.addCleanup(patch.stop)

    def make_case(self, environment=None):
        """Resolve a temporary admitted baseline into one immutable concrete case."""
        if environment is not None:
            controller.atomic_json(self.sentinel, {**self.ready, "reference_environment": environment})
        ready = controller.ready_specification(self.run)
        return controller.immutable_case(ideas.INITIAL_IDEAS[6], "h100", 1, ready)

    def test_missing_environment_keeps_original_case_contents(self):
        """An omitted or empty baseline retains the original case and digest exactly."""
        missing = self.make_case()
        explicit = self.make_case({})
        self.assertEqual(missing, explicit)
        self.assertNotIn("reference_environment", missing)
        self.assertEqual(controller.ready_specification(self.run)["reference_environment"], {})

    def test_environment_is_copied_and_enters_the_case_digest(self):
        """Changing a baseline setting changes its identity without sharing a mutable mapping."""
        baseline = {"cache_room_backgrounds": True, "batch_layout": "nested"}
        case = self.make_case(baseline)
        self.assertEqual(case["reference_environment"], baseline)
        changed = self.make_case({"cache_room_backgrounds": False, "batch_layout": "nested"})
        self.assertNotEqual(case["specification_digest"], changed["specification_digest"])
        baseline["batch_layout"] = "flat"
        self.assertEqual(case["reference_environment"]["batch_layout"], "nested")

    def test_protocol_and_unknown_keys_are_rejected_even_at_original_values(self):
        """A performance baseline cannot restate or change game or observation protocol."""
        protocol = {"num_envs": 128, "frame_skip": 4, "max_episode_steps": 108000,
                    "noop_max": 30, "episodic_life": True, "sticky_action_probability": 0.25,
                    "grayscale_coefficients": [0.2989, 0.5870, 0.1140],
                    "resize_method": "area", "invented_setting": True}
        for name, value in protocol.items():
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "only environment performance"):
                self.make_case({name: value})

    def test_performance_values_have_strict_types_and_source_support(self):
        """Reject integer booleans, invalid layouts, and absent constructor capabilities."""
        for value in (None, [], {3: True}, {"cache_room_backgrounds": 1}, {"frame_skip_unroll": True},
                      {"frame_skip_unroll": 0}, {"frame_skip_unroll": float("inf")},
                      {"batch_layout": "other"}, {"stack_update": "other"}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                controller.reference_environment_settings(value, self.source)
        with mock.patch.object(controller, "source_capabilities", return_value={"environment": set()}):
            with self.assertRaisesRegex(ValueError, "not implemented in the pinned source"):
                controller.reference_environment_settings({"cache_room_backgrounds": True}, self.source)
        accepted = {"frame_skip_unroll": 4, "stack_update": "circular",
                    "fuse_pixel_preprocessing": True}
        self.assertEqual(controller.reference_environment_settings(accepted, self.source), accepted)

    def test_changed_authorization_fails_before_any_numerical_import(self):
        """A previously valid pending case cannot silently adopt another admitted baseline."""
        case = self.make_case({"cache_room_backgrounds": True})
        controller.atomic_json(self.sentinel, {**self.ready, "reference_environment": {
            "cache_room_backgrounds": False}})
        with mock.patch.object(run_case, "wait_for_recorded_release", return_value={}), \
                mock.patch.object(run_case, "hardware_record") as hardware:
            with self.assertRaisesRegex(ValueError, "current authorization"):
                run_case.run_case(self.run, case, 1, "synthetic-token")
            hardware.assert_not_called()
        self.assertNotIn("jax", sys.modules)

    def test_admitted_environment_reaches_the_paired_benchmark(self):
        """Exercise the runner boundary with fake numerical modules and a captured call."""
        environment = {"cache_room_backgrounds": True, "frame_skip_unroll": 4}
        case = self.make_case(environment)
        numerical_modules = {name: types.ModuleType(name) for name in (
            "jax", "exploration_platform", "exploration_platform.atari_pixels",
            "exploration_platform.atari_pixels.config", "paired_benchmark")}
        numerical_modules["jax"].config = types.SimpleNamespace(update=mock.Mock())
        numerical_modules["exploration_platform.atari_pixels.config"].PixelPPOConfig = (
            lambda **values: types.SimpleNamespace(**values))
        paired = numerical_modules["paired_benchmark"]
        paired.CorrectnessMismatch = type("SyntheticCorrectnessMismatch", (ValueError,), {})
        paired.run_paired_benchmark = mock.Mock(return_value={
            "hardware_class": "h100", "source_commit": self.commit,
            "decision": "accepted", "measurement_valid": True})
        before = set(sys.modules)
        with mock.patch.dict(sys.modules, numerical_modules), \
                mock.patch.object(run_case, "wait_for_recorded_release", return_value={}), \
                mock.patch.object(run_case, "hardware_record", return_value={"pid": os.getpid()}), \
                mock.patch.object(run_case, "asdict", side_effect=lambda value: vars(value)):
            result = run_case.run_case(self.run, case, 1, "synthetic-token")
        self.assertTrue(result["case_completed"])
        self.assertEqual(paired.run_paired_benchmark.call_args.kwargs["reference_environment"], environment)
        self.assertEqual(paired.run_paired_benchmark.call_args.kwargs["reference_execution"],
                         case["reference_execution"])
        self.assertEqual(set(sys.modules) - before, set())
        self.assertEqual(json.loads((self.run / "data" / case["unit_id"] / "attempts/attempt-0001/case.json")
                                    .read_text())["reference_environment"], environment)


if __name__ == "__main__":
    unittest.main()
