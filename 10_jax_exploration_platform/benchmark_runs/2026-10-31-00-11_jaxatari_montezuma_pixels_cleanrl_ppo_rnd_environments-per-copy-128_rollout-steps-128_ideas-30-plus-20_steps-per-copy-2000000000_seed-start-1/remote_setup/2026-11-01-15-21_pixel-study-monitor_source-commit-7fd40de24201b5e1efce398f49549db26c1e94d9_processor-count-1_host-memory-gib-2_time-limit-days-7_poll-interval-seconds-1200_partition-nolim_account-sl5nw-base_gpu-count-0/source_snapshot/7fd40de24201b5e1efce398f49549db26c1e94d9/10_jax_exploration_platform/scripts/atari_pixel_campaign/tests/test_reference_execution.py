"""Check future execution baselines with syntax trees and standard-library metadata only."""

from dataclasses import replace
import hashlib
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
import controller
import ideas
import run_case
import test_admit_measurements as admission_fixtures
import test_reference_environment as environment_fixtures


class ReferenceExecutionTests(unittest.TestCase):
    """Preserve the initial protocol while carrying accepted execution changes into both arms."""

    def setUp(self):
        """Reuse complete synthetic admission files and mock only source identity checks."""
        self.fixture = environment_fixtures.ReferenceEnvironmentTests("runTest")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.source = self.fixture.source

    def make_case(self, baseline, entry=None):
        """Build a case from a temporary admission without touching the live campaign."""
        controller.atomic_json(self.fixture.sentinel, {**self.fixture.ready,
                                                      "reference_execution": baseline})
        ready = controller.ready_specification(self.fixture.run)
        return controller.immutable_case(entry or ideas.INITIAL_IDEAS[6], "h100", 1, ready)

    def test_empty_baseline_preserves_prechange_case_digests_and_reporting_control(self):
        """Use fixed metadata digests captured from committed d7071a1 before these changes."""
        ready = {"campaign_run_dir": "/synthetic_campaign", "source_root": "/synthetic_source",
            "source_commit": "d28e6735e79788f618acb0ebfde8609db1f776af",
            "reference_configuration": dict(controller.REFERENCE_ALGORITHM)}
        expected = {1: "76ec8a2f01d01a2ea9846bd91a062c3585ce747da9288ce111af22101958a0f2",
                    27: "8be72882f8c75fa80d5787d890cd141d673728de7dabd004d6968f0ba905b9c7",
                    29: "dbe49c97d7289c8c58918232cde1afbfdb8dd631598fdba9c06918d4ffcf9548"}
        for position, digest in expected.items():
            entry = ideas.INITIAL_IDEAS[position - 1]
            missing = controller.immutable_case(entry, "h100", 1, ready)
            explicit = controller.immutable_case(entry, "h100", 1, {**ready, "reference_execution": {}})
            self.assertEqual(missing, explicit)
            self.assertEqual(missing["specification_digest"], digest)
            self.assertEqual(missing["reference_execution"], entry.reference_execution)
            self.assertEqual(missing["candidate"]["execution"], entry.execution)
            self.assertNotIn("reference_execution_baseline", missing)

    def test_baseline_merges_both_arms_and_enters_the_digest(self):
        """Retain a group baseline when the candidate changes only learner or environment work."""
        baseline = {"iterations_per_call": 5}
        case = self.make_case(baseline)
        expected = {**controller.execution_defaults(self.source), **baseline}
        self.assertEqual(case["reference_execution"], expected)
        self.assertEqual(case["candidate"]["execution"], expected)
        self.assertEqual(case["reference_execution_baseline"], baseline)
        self.assertEqual(case["iterations"], 5)
        changed = self.make_case({"iterations_per_call": 2})
        self.assertNotEqual(case["specification_digest"], changed["specification_digest"])
        baseline["iterations_per_call"] = 9
        self.assertEqual(case["reference_execution_baseline"]["iterations_per_call"], 5)
        self.assertEqual(case["configuration"], changed["configuration"])

    def test_explicit_controls_override_independently_and_set_both_measurement_windows(self):
        """Use the least common window from resolved arms, including inherited report settings."""
        entry = replace(ideas.INITIAL_IDEAS[6], execution={"iterations_per_call": 2})
        grouped = self.make_case({"iterations_per_call": 5}, entry)
        self.assertEqual(grouped["reference_execution"]["iterations_per_call"], 5)
        self.assertEqual(grouped["candidate"]["execution"]["iterations_per_call"], 2)
        self.assertEqual(grouped["iterations"], 10)
        reporting = replace(entry, reference_execution={"report_every_iterations": 3},
                            execution={"report_every_iterations": 4})
        reported = self.make_case({"report_readout": "scheduled", "report_every_iterations": 10}, reporting)
        self.assertEqual(reported["iterations"], 12)
        self.assertEqual(reported["reference_execution"]["report_every_iterations"], 3)
        self.assertEqual(reported["candidate"]["execution"]["report_every_iterations"], 4)
        ref_only = replace(entry, reference_execution={"report_readout": "per_update"}, execution={})
        independent = self.make_case({"report_every_iterations": 20}, ref_only)
        self.assertEqual(independent["reference_execution"]["report_readout"], "per_update")
        self.assertEqual(independent["candidate"]["execution"]["report_readout"], "none")
        self.assertEqual(independent["iterations"], 20)

    def test_bad_values_and_conflicting_combinations_fail_before_case_creation(self):
        """Reject algorithm/runtime options, bool counts, unsupported modes and conflicting arms."""
        for value in (None, [], {4: "none"}, {"learning_rate": 1e-4}, {"deterministic_operations": False},
                      {"iterations_per_call": True}, {"report_every_iterations": 1.5},
                      {"iterations_per_call": 0}, {"report_readout": "other"},
                      {"compilation_mode": "other"},
                      {"compilation_mode": "separate_phases", "iterations_per_call": 5},
                      {"report_readout": "scheduled", "iterations_per_call": 5}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                controller.reference_execution_settings(value, self.source)
        for execution in ({"compilation_mode": "separate_phases"}, {"report_readout": "scheduled"}):
            entry = replace(ideas.INITIAL_IDEAS[6], execution=execution)
            with self.subTest(execution=execution), self.assertRaises(ValueError):
                self.make_case({"iterations_per_call": 5}, entry)
        self.assertFalse((self.fixture.run / "queue/specifications").exists())

    def test_ast_inspection_requires_declared_and_consumed_source_options(self):
        """A declared unused field cannot be admitted as an implemented optimization."""
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)
            target = source / "10_jax_exploration_platform/scripts/atari_pixel_campaign/paired_benchmark.py"
            target.parent.mkdir(parents=True)
            target.write_text("raise RuntimeError('must never import numerical source')\n"
                + "DEFAULT_EXECUTION = " + repr(controller.execution_defaults(self.source))
                + "\nclass Executor:\n    def run(self):\n        return self.options['iterations_per_call']\n")
            with self.assertRaisesRegex(ValueError, "does not implement"):
                controller.reference_execution_settings({"iterations_per_call": 5}, source)
        self.assertNotIn("jax", sys.modules)

    def test_changed_admission_fails_before_hardware_or_numerical_imports(self):
        """An old pending case cannot adopt a different global execution baseline."""
        case = self.make_case({"iterations_per_call": 5})
        controller.atomic_json(self.fixture.sentinel, {**self.fixture.ready,
                                                      "reference_execution": {"iterations_per_call": 2}})
        with mock.patch.object(run_case, "wait_for_recorded_release", return_value={}), \
                mock.patch.object(run_case, "hardware_record") as hardware:
            with self.assertRaisesRegex(ValueError, "reference execution.*current authorization"):
                run_case.run_case(self.fixture.run, case, 1, "synthetic-token")
            hardware.assert_not_called()
        self.assertNotIn("jax", sys.modules)

    def test_resolved_arms_reach_existing_benchmark_boundary(self):
        """Capture both execution dictionaries using fake numerical modules and no actual arrays."""
        case = self.make_case({"iterations_per_call": 5})
        modules = {name: types.ModuleType(name) for name in ("jax", "exploration_platform",
            "exploration_platform.atari_pixels", "exploration_platform.atari_pixels.config", "paired_benchmark")}
        modules["jax"].config = types.SimpleNamespace(update=mock.Mock())
        modules["exploration_platform.atari_pixels.config"].PixelPPOConfig = (
            lambda **values: types.SimpleNamespace(**values))
        paired = modules["paired_benchmark"]
        paired.CorrectnessMismatch = type("SyntheticMismatch", (ValueError,), {})
        paired.run_paired_benchmark = mock.Mock(return_value={"hardware_class": "h100",
            "source_commit": self.fixture.commit, "decision": "accepted", "measurement_valid": True})
        with mock.patch.dict(sys.modules, modules), \
                mock.patch.object(run_case, "wait_for_recorded_release", return_value={}), \
                mock.patch.object(run_case, "hardware_record", return_value={"pid": os.getpid()}), \
                mock.patch.object(run_case, "asdict", side_effect=lambda value: vars(value)):
            run_case.run_case(self.fixture.run, case, 1, "synthetic-token")
        self.assertEqual(paired.run_paired_benchmark.call_args.args[3], case["candidate"]["execution"])
        self.assertEqual(paired.run_paired_benchmark.call_args.kwargs["reference_execution"],
                         case["reference_execution"])
        self.assertEqual(paired.run_paired_benchmark.call_args.kwargs["iterations"], 5)

    def test_admission_records_optional_configuration_and_exact_file_evidence(self):
        """The optional future input preserves a file hash without revising an existing admission."""
        fixture = admission_fixtures.AdmissionTests("runTest")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.arguments.source_root = self.source
        path = fixture.write("execution.json", {"iterations_per_call": 5})
        fixture.arguments.reference_execution_configuration = path
        fixture.run_admission()
        sentinel = controller.read_json(fixture.run_dir / controller.SENTINEL_NAME)
        self.assertEqual(sentinel["reference_execution"], {"iterations_per_call": 5})
        self.assertEqual(sentinel["reference_execution_source"]["sha256"],
                         hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertEqual(sentinel["reference_execution_source"]["path"], str(path.resolve()))
        before = (fixture.run_dir / controller.SENTINEL_NAME).read_bytes()
        fixture.write(path, {"iterations_per_call": 2})
        with self.assertRaisesRegex(ValueError, "immutable admission"):
            fixture.run_admission()
        self.assertEqual((fixture.run_dir / controller.SENTINEL_NAME).read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
