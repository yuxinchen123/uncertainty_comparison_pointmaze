"""Test new qualification metadata and actual own-CPU startup; no GPU or scientific units."""
import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import types
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("profile_target_qualification_tests", SCRIPTS / "profile_target_qualification.py")
Q = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(Q)


class TargetQualificationMetadataTests(unittest.TestCase):
    """Use explicitly synthetic metadata premises, never claim actual target qualification."""
    def setUp(self):
        """Create an isolated ordinary parent/reference fixture with full-size count fields."""
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.configuration = {"learner": {"n_copies": 2, "n_envs": 128, "num_steps": 128,
            "update_epochs": 4, "num_minibatches": 4, "observation_warmup_iterations": 50,
            "total_timesteps": 2000000000}, "environment": {"cache_room_backgrounds": True,
            "render_final_two_only": True}, "execution": {"iterations_per_call": 1}}
        self.plan = {"kind": Q.KIND, "source_root": "/explicit-synthetic-source", "source_commit": Q.SOURCE,
            "campaign_run_dir": str(self.root), "unit_id": "synthetic-profile-capture",
            "core": {"synthetic_premise": True}, "paired_source": {"synthetic_premise": True},
            "helpers": {"synthetic_premise": True}, "configuration": self.configuration,
            "runtime_environment": {"JAX_DEFAULT_MATMUL_PRECISION": "highest", "JAX_ENABLE_X64": "true",
                "XLA_FLAGS": "--xla_gpu_exclude_nondeterministic_ops=true"}, "asset_record": {"synthetic_premise": True},
            "hardware_class": "jaguar03_rtx_a4500", "copy_seeds": [1, 2], "python": "/explicit-synthetic/bin/python",
            "profile_updates": 1, "profile_tool": {"synthetic_premise": True},
            "configured_isolation": {"consumer": "component_profile", "synthetic_premise": True}}
        original = copy.deepcopy(self.plan)
        original["configuration"]["learner"]["n_copies"] = 1
        original["configuration"]["environment"] = {"cache_room_backgrounds": False, "render_final_two_only": False}
        original["reference_configuration"] = copy.deepcopy(original["configuration"])
        original.update({key: {"synthetic_premise": True} for key in (
            "qualification_sources", "cleanrl_reference", "qualification_software", "renderer_contract")})
        self.original_ref = self.write("original_gate_plan.json", original)
        self.parent = {"kind": Q.CONTROLLER_KIND, "protocol": Q.PROTOCOL,
            "profile_unit": str(self.root / "data" / self.plan["unit_id"]), "campaign_run_dir": str(self.root),
            "source_commit": Q.SOURCE, "target_fields": Q.target_fields(self.plan),
            "original_qualification_plan": self.original_ref, "scientific_credit": 0, "admission_granted": False,
            "qualification_time_budget": dict(Q.QUALIFICATION_BUDGET), "controller_maximum_seconds": 10800}
        self.parent_ref = self.write("controller_plan.json", self.parent)
        self.plan["profile_target_qualification"] = {"schema_version": 1, "protocol": Q.PROTOCOL,
            "parent": self.parent_ref}

    def write(self, name, value):
        """Store one exact owned temporary JSON fixture reference."""
        path = self.root / name
        path.write_bytes(Q.encoded(value))
        return Q.reference(path)

    def test_distinct_parent_kind_and_full_size_counts_are_required(self):
        """Full typed counts pass; Boolean seeds/counts and an ordinary kind cannot qualify."""
        self.assertEqual(Q.contract(self.plan), self.parent)
        for change in ("seed", "count", "kind", "runtime", "parent"):
            plan = copy.deepcopy(self.plan)
            if change == "seed":plan["copy_seeds"] = [True, 2]
            elif change == "count":plan["configuration"]["learner"]["n_envs"] = True
            elif change == "kind":plan["kind"] = "selected_configuration_profile"
            elif change == "runtime":plan["runtime_environment"]["JAX_ENABLE_X64"] = "false"
            else:plan["unit_id"] = "different-unit"
            with self.assertRaises(ValueError):Q.contract(plan)

    def test_qualification_never_borrows_an_ordinary_admission(self):
        """A qualification parent cannot be used as a launch admission or mixed purpose."""
        plan = copy.deepcopy(self.plan);plan["profile_admission_inputs"] = {}
        with self.assertRaisesRegex(ValueError, "cannot borrow"):Q.contract(plan)
        ordinary = copy.deepcopy(self.plan);ordinary.pop("profile_target_qualification");ordinary["kind"] = "selected_configuration_profile"
        with self.assertRaisesRegex(ValueError, "remain closed"):Q.authorize(ordinary)
        with self.assertRaisesRegex(ValueError, "markers"):Q.authorize({**self.plan, "kind": "selected_configuration_profile"})

    def test_original_gate_projection_preserves_one_copy_and_leaves_profile_unchanged(self):
        """The unchanged original gate runner gets one copy while actual profile work remains two copies."""
        before = copy.deepcopy(self.plan)
        original = Q.original_gate_plan(self.plan)
        self.assertEqual(original["configuration"]["learner"]["n_copies"], 1)
        self.assertEqual(original["copy_seeds"], [1])
        self.assertEqual(original["configured_isolation"]["consumer"], "selected_measurement")
        self.assertEqual(original["configuration"], original["reference_configuration"])
        self.assertNotIn("profile_target_qualification", original)
        self.assertEqual(self.plan, before)

    def test_reference_derivation_refuses_oversized_and_symlink_before_open(self):
        """An unknown large or indirect metadata input is never read as a proof."""
        oversized = self.root / "too_large.json"
        with oversized.open("wb") as stream:stream.truncate(Q.MAXIMUM + 1)
        linked = self.root / "linked.json";linked.symlink_to(self.parent_ref["path"])
        for path in (oversized, linked):
            with patch.object(Q.os, "open", side_effect=AssertionError("input must not open")) as opened:
                with self.assertRaises(ValueError):Q.reference(path)
            opened.assert_not_called()
        duplicated = self.root / "duplicated.json";duplicated.write_text('{"x":1,"x":2}')
        with self.assertRaisesRegex(ValueError, "duplicate"):Q.document(duplicated)

    def capture_fixture(self):
        """Describe synthetic complete counters with an actual CPU PID, not numerical execution."""
        now = time.time()
        captured = {"normalization_warmup_iterations": 50, "compilation_warmup": [{"updates": 1}, {"updates": 1}],
            "profile_start_iteration": 2, "profile_end_iteration": 3,
            "compiled_signatures_before": [1], "compiled_signatures_after": [1],
            "execution_identity": {"hostname": "explicit-synthetic", "pid": os.getpid(),
                "process_start_time": "synthetic-birth", "started_at_seconds": now - 3, "finished_at_seconds": now - 2}}
        original = {"reference_iteration": 1, "reference_optimizer_step": 16, "executed_work": {
            "full_reference_warmup_transitions": 819200, "full_reference_training_transitions": 16384,
            "full_reference_updates": 1, "full_reference_optimizer_minibatch_updates": 16,
            "whole_job_transition_total_known": False}}
        return captured, original, now - 1, now

    def test_actual_counter_fields_never_multiply_original_fixture_work(self):
        """Two profile copies do not turn one original reference update into two reference updates."""
        captured, original, start, finish = self.capture_fixture()
        result = Q.controlled_work(self.plan, captured, original, start, finish)
        self.assertEqual(result["profile_capture"]["copies"], 2)
        self.assertEqual(result["profile_capture"]["compile_warmup_updates"], 2)
        self.assertEqual(result["original_reference"]["copies"], 1)
        self.assertEqual(result["original_reference"]["executed_work"]["training_transitions"], 16384)
        self.assertEqual(result["test_fixture_work"], "unknown")
        self.assertFalse(result["whole_job_transition_total_known"])
        self.assertEqual(result["original_gate_raw_executed_work"], original["executed_work"])
        backend_spec = importlib.util.spec_from_file_location("real_completed_work_interface", SCRIPTS / "profile_qualification_backend.py")
        backend = importlib.util.module_from_spec(backend_spec);backend_spec.loader.exec_module(backend)
        self.assertEqual(backend.completed_work(self.plan, result), "full_original_reference_and_profile_capture")
        for key in ("profile_start_iteration", "profile_end_iteration", "normalization_warmup_iterations"):
            changed = copy.deepcopy(captured);changed[key] += 1
            with self.assertRaises(ValueError):Q.controlled_work(self.plan, changed, original, start, finish)
        changed = copy.deepcopy(captured);changed["compilation_warmup"][0]["updates"] = True
        with self.assertRaises(ValueError):Q.controlled_work(self.plan, changed, original, start, finish)

    def test_missing_fourth_context_refuses_before_historical_reader(self):
        """Three genuine-shaped metadata rows cannot replace H100 two-copy acceptance."""
        proof = {"kind": "reviewed_four_context_configured_profile_prerequisites", "source_commit": Q.SOURCE,
            "campaign_run_dir": str(self.root), "operational_admission": False,
            "contexts": [{"hardware_class": h, "copies": c} for h, c in (
                ("h100", 1), ("jaguar03_rtx_a4500", 1), ("jaguar03_rtx_a4500", 2))]}
        proof_ref = self.write("incomplete_four_context_proof.json", proof)
        review_ref = self.write("synthetic_proof_review.json", {"kind": "independent_reviewed_four_context_configured_profile_prerequisites",
            "proof": proof_ref, "approved": True, "passed": True, "independent_of_producer": True,
            "all_four_raw_reviews_verified": True, "exact_historical_consumer_namespaces_verified": True,
            "unresolved_anomalies": [], "admission_granted": False})
        class Files:
            """Read temporary bounded fixtures only."""
            def document(self, value):return Q.checked(value)
        reader = types.SimpleNamespace(PROTOCOL={"kind": proof["kind"]}, MetadataFiles=Files,
            CONTEXTS={(h, c) for h in ("h100", "jaguar03_rtx_a4500") for c in (1, 2)})
        with patch.object(Q, "adjacent", return_value=reader):
            with self.assertRaisesRegex(ValueError, "missing H100 two-copy"):
                Q.require_four_contexts(self.plan, proof_ref, review_ref)

    def test_unreviewed_or_partial_result_never_writes_an_admission(self):
        """Process return zero or a partial profile cannot create a scientific admission."""
        result = self.write("partial_target.json", {"kind": Q.KIND, "completed": False,
            "measurement_valid": False, "scientific_credit": 0})
        review = self.write("unapproved_target_review.json", {"kind": Q.REVIEW_KIND, "approved": False})
        output = self.root / "admission.json"
        with self.assertRaises(ValueError):Q.create_admission(result["path"], review["path"], output)
        self.assertFalse(output.exists())

    def test_retained_startup_custody_waits_through_ordinary_errors_without_signals(self):
        """A broken wait observation never releases caller custody as successful cleanup."""
        class Child:
            """Provide typed real-Popen-like wait outcomes for a pure lifecycle refusal fixture."""
            pid = 123
            def __init__(self):self.calls = 0
            def wait(self, timeout):
                self.calls += 1
                if self.calls == 1:raise OSError("explicit synthetic wait observation failure")
                return 0
        child = Child()
        with patch.object(Q.time, "sleep", return_value=None):result = Q.retain_startup_custody(child)
        self.assertEqual(result["returncode"], 0)
        self.assertFalse(result["signal_sent"])
        self.assertEqual(child.calls, 2)
        self.assertEqual(result["ordinary_observer_errors"][0]["operation"], "exact_wait")

    def test_actual_own_cpu_child_is_recorded_and_waited_without_gpu_work(self):
        """Run one isolated standard-library CPU child and verify real UID/birth/argv/affinity records."""
        spec = importlib.util.spec_from_file_location("profile_target_cpu_process_reader", SCRIPTS / "direct_common_timing_controller.py")
        control = importlib.util.module_from_spec(spec);spec.loader.exec_module(control)
        command = ["/usr/bin/python3", "-I", "-S", "-B", "-c", "import time; time.sleep(.25)"]
        before = sorted(os.sched_getaffinity(0));affinity = before[:1]
        try:
            os.sched_setaffinity(0, affinity)
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        finally:os.sched_setaffinity(0, before)
        try:observed = Q.exact_child_startup(control, child, command, affinity, self.root)
        finally:self.assertEqual(child.wait(timeout=3), 0)
        self.assertEqual(observed["pid"], child.pid)
        self.assertEqual(observed["parent_pid"], os.getpid())
        receipt = Q.document(self.root / "profile_target_child_startup_observations.json")
        self.assertTrue(receipt["exact_command_ready"])
        self.assertIsNone(receipt["failure"])
        self.assertFalse(receipt["controller_holds_standing_descriptor"])
        self.assertEqual(sorted(os.sched_getaffinity(0)), before)


if __name__ == "__main__":
    unittest.main()
