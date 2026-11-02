"""Exercise metadata prerequisite checks without numerical or device operations."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
import common_confirmation as common
import measure_selected as measurement


class CommonConfirmationTests(unittest.TestCase):
    """Keep current common checks separate from historical and final protocols."""

    def setUp(self):
        """Make one minimal explicit intermediate configuration and immutable inputs."""
        self.temporary = tempfile.TemporaryDirectory(); self.addCleanup(self.temporary.cleanup)
        self.unit = Path(self.temporary.name)
        self.plan = {"kind": "selected_configuration_measurement", "mode": "confirmation",
            "hardware_class": "jaguar03_rtx_a4500", "gpu_uuid": None, "configured_isolation": {"profile": {}},
            "rounds": 7, "minimum_updates_per_round": 10, "iterations": 10, "copy_seeds": [1],
            "configuration": {"learner": {"n_copies": 1}, "environment": {"cache_room_backgrounds": True,
                "render_final_two_only": True}, "execution": {}},
            "reference_configuration": {"learner": {"n_copies": 1}, "environment": {"cache_room_backgrounds": False,
                "render_final_two_only": False}, "execution": {}},
            "helpers": {"source_commit": "a" * 40, "files": {common.PREFIX + common.NAME:
                hashlib.sha256(Path(common.__file__).read_bytes()).hexdigest()}},
            "common_correctness_inputs": {"result": {"path": str(self.unit / "prior.json"), "sha256": "b" * 64},
                "review": {"path": str(self.unit / "review.json"), "sha256": "c" * 64}}}
        (self.unit / "measurement_plan.json").write_bytes(common.encoded(self.plan))

    def proof(self, raw=None):
        """Return a synthetic successful child record with this actual test parent."""
        raw = common.encoded(self.plan) if raw is None else raw
        return {"schema_version": 1, "kind": "isolated_common_correctness_validation", "passed": True,
            "plan_sha256": common.digest(raw), "common_correctness_inputs": self.plan["common_correctness_inputs"],
            "parent_pid": os.getpid(), "validation_pid": 12, "finished_at_seconds": 1., "numerical_imports": [], "gpu_operations": 0}

    def completed(self, proof=None):
        """Stand in only for the separately tested metadata child process."""
        return subprocess.CompletedProcess([], 0, json.dumps(proof or self.proof()).encode(), b"")

    def test_intermediate_contract_accepts_both_explicit_copy_counts(self):
        """The two accepted copy configurations retain their exact seed lists."""
        common.contract(self.plan)
        self.plan["configuration"]["learner"]["n_copies"] = 2
        self.plan["reference_configuration"]["learner"]["n_copies"] = 2
        self.plan["copy_seeds"] = [1, 2]
        common.contract(self.plan)

    def test_final_direct_capacity_and_wrong_mode_are_rejected(self):
        """No opt-in field can authorize a different or later protocol."""
        for key in ("direct_binding", "target_qualification", "final_confirmation_protocol", "completed_study",
                    "device_memory_limit_protocol", "fixed_total_scaling", "scaling_protocol"):
            with self.subTest(key=key), self.assertRaises(ValueError):common.contract({**self.plan, key: {}})
        for mode in ("correctness", "packing-point", "learning-check"):
            with self.subTest(mode=mode), self.assertRaises(ValueError):common.contract({**self.plan, "mode": mode})

    def test_window_types_and_incomplete_prerequisites_are_rejected(self):
        """Boolean counters, a reduced window and one missing record cannot pass."""
        for key, wrong in (("rounds", True), ("rounds", 11), ("iterations", 1), ("minimum_updates_per_round", 20),
                ("common_correctness_inputs", {"result": {}})):
            with self.subTest(key=key), self.assertRaises(ValueError):common.contract({**self.plan, key: wrong})

    def test_equal_or_extra_candidate_change_and_wrong_seeds_are_rejected(self):
        """Only the independently checked two environment changes are permitted."""
        equal = {**self.plan, "configuration": self.plan["reference_configuration"]}
        with self.assertRaises(ValueError):common.contract(equal)
        changed = copy.deepcopy(self.plan); changed["configuration"]["learner"]["learning_rate"] = 1
        changed["reference_configuration"]["learner"]["learning_rate"] = 2
        with self.assertRaises(ValueError):common.contract(changed)
        with self.assertRaises(ValueError):common.contract({**self.plan, "copy_seeds": [True]})

    def test_staged_helper_bytes_and_reference_hashes_are_required(self):
        """A different helper or changed saved bytes do not substitute for the frozen records."""
        changed = copy.deepcopy(self.plan); changed["helpers"]["files"][common.PREFIX + common.NAME] = "0" * 64
        with self.assertRaises(ValueError):common.contract(changed)
        p = self.unit / "reference.json"; p.write_text('{"value":1}')
        record = common.reference(p); self.assertEqual(common.checked_reference(record), {"value": 1})
        p.write_text('{"value":2}')
        with self.assertRaises(ValueError):common.checked_reference(record)

    def test_child_request_leaves_parent_paths_and_public_modules_unchanged(self):
        """The parent dispatches only the declared no-site interpreter and changes no import paths."""
        before = list(sys.path); ideas = sys.modules.get("ideas")
        with patch.object(common.subprocess, "run", return_value=self.completed()) as run:
            common.request(self.plan)
        command = run.call_args.args[0]
        self.assertEqual(command, ["/usr/bin/python3", "-I", "-S", "-B", str(Path(common.__file__).absolute()), "--plan-stdin"])
        self.assertEqual(before, sys.path); self.assertIs(sys.modules.get("ideas"), ideas)

    def test_different_parent_raw_plan_and_boolean_gpu_claim_are_rejected(self):
        """A successful child response must describe this exact parent and plan."""
        for key, value in (("parent_pid", True), ("parent_pid", os.getpid() + 1), ("plan_sha256", "0" * 64), ("gpu_operations", False)):
            with self.subTest(key=key), patch.object(common.subprocess, "run", return_value=self.completed({**self.proof(), key: value})):
                with self.assertRaises(ValueError):common.request(self.plan)

    def test_success_captures_raw_child_before_publishing_proof_and_cannot_repeat(self):
        """Keep one durable actual validation attempt and its exact raw successful return."""
        with patch.object(common.subprocess, "run", return_value=self.completed()) as run:
            common.before_compute(self.unit, self.plan)
            binding = common.result_binding(self.unit, self.plan)
            self.assertEqual(binding["common_correctness_inputs"], self.plan["common_correctness_inputs"])
            self.assertTrue((self.unit / "common_correctness_validation_stdout.raw").exists())
            with self.assertRaises(ValueError):common.before_compute(self.unit, self.plan)
            self.assertEqual(run.call_count, 1)

    def test_failed_and_malformed_child_return_preserves_raw_without_success_proof(self):
        """Retain both nonzero and invalid successful output before refusing the worker."""
        for value in (subprocess.CompletedProcess([], 1, b"partial", b"failed"), subprocess.CompletedProcess([], 0, b"invalid JSON", b"")):
            with tempfile.TemporaryDirectory() as tmp:
                unit = Path(tmp); (unit / "measurement_plan.json").write_bytes(common.encoded(self.plan))
                with patch.object(common.subprocess, "run", return_value=value):
                    with self.assertRaises((ValueError, json.JSONDecodeError)):common.before_compute(unit, self.plan)
                self.assertEqual((unit / "common_correctness_validation_stdout.raw").read_bytes(), value.stdout)
                self.assertFalse((unit / common.PROOF_NAME).exists())
                with self.assertRaises(ValueError):common.before_compute(unit, self.plan)

    def test_timeout_preserves_partial_output_and_attempt(self):
        """A timed-out child never becomes a successful prerequisite or starts a repeat."""
        with patch.object(common.subprocess, "run", side_effect=subprocess.TimeoutExpired('synthetic', 1, output=b'partial', stderr=b'error')):
            with self.assertRaises(subprocess.TimeoutExpired):common.before_compute(self.unit, self.plan)
        self.assertEqual((self.unit / 'common_correctness_validation_stdout.raw').read_bytes(), b'partial')
        self.assertIsNone(common.read(self.unit / 'common_correctness_validation_return.json')[0]['returncode'])
        self.assertFalse((self.unit / common.PROOF_NAME).exists())

    def test_saved_proof_requires_same_actual_parent_and_precedes_numeric_work(self):
        """The saved result cannot claim another worker's or later validation."""
        with patch.object(common.subprocess, 'run', return_value=self.completed()):common.before_compute(self.unit, self.plan)
        result = {**common.result_binding(self.unit, self.plan), 'execution_identity': {'pid': os.getpid(), 'started_at_seconds': 2.}}
        with patch.object(common, 'request') as request:common.verify_saved(self.unit, self.plan, result)
        request.assert_called_once()
        for changed in ({'pid': os.getpid() + 1, 'started_at_seconds': 2.}, {'pid': os.getpid(), 'started_at_seconds': .5}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):common.verify_saved(self.unit, self.plan, {**result, 'execution_identity': changed})

    def test_default_direct_and_new_helper_closures_stay_separate(self):
        """Add only the new configured dependency while retaining all existing helper declarations."""
        baseline = measurement.required_helpers(True)
        self.assertEqual(measurement.required_helpers(True, False, True), (*baseline, common.NAME))
        self.assertNotIn(common.NAME, measurement.required_helpers())
        self.assertNotIn(common.NAME, measurement.required_helpers(False, True))
        with self.assertRaises(ValueError):measurement.required_helpers(False, True, True)

    def test_cli_requires_both_actual_records_and_refuses_final_or_prepared_overrides(self):
        """Preparation options cannot silently change the frozen protocol."""
        for extra in (["--common-correctness-record", "/missing"],
                ["--prepared-unit", "/missing", "--common-correctness-record", "/missing", "--common-correctness-review", "/missing"],
                ["--mode", "confirmation", "--final-confirmation", "--configured-isolation-profile", "/missing", "--common-correctness-record", "/missing", "--common-correctness-review", "/missing"]):
            with self.subTest(extra=extra), self.assertRaises(SystemExit):measurement.parse_arguments(extra)


if __name__ == '__main__':
    unittest.main()
