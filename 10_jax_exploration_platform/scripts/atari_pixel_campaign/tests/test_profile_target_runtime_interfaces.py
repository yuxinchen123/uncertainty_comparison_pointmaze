"""Exercise the reproduced runtime interfaces with explicit CPU-only metadata premises."""
import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1]
PREFIX = "10_jax_exploration_platform/scripts/atari_pixel_campaign/"


def load(name, tests=False):
    """Load exact owned source or a declared synthetic fixture; never initialize a GPU/native provider."""
    path = SCRIPTS / ("tests/" if tests else "") / (name + ".py")
    spec = importlib.util.spec_from_file_location("runtime_interface_" + name, path)
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


class RuntimeInterfaceTests(unittest.TestCase):
    """Keep source/ownership/native initialization outside the explicitly stubbed metadata premises."""
    def test_all_three_h_entrypoints_pass_one_actual_deadline_to_real_runtime(self):
        """Reach the unchanged provider boundary only after the actual Runtime accepts the passed budget."""
        profile, process = load("profile_selected"), load("profile_target_process")
        with tempfile.TemporaryDirectory() as temporary:
            unit = Path(temporary);(unit / "code").mkdir()
            plan = {"kind": process.QUALIFICATION, "python": "/explicit-fixture/bin/python",
                "diagnostic_time_budget": dict(process.QUALIFICATION_BUDGET), "hardware_class": "h100",
                "helpers": {"files": {PREFIX + process.NAME: hashlib.sha256(Path(process.__file__).read_bytes()).hexdigest()}}}
            (unit / "measurement_plan.json").write_text(json.dumps(plan))
            commands = {"producer": process.producer_command(unit, plan), "waiter": ["/explicit-fixture/bin/python", "waiter"],
                "capture": ["/explicit-fixture/bin/python", "capture"]}
            current_role = ["producer"]
            metadata = types.SimpleNamespace(stable=lambda value: value, validate_snapshot=lambda *args: None,
                publish=lambda *args: None, receipt=lambda path: {"path": str(path), "sha256": "a" * 64})
            facade = types.SimpleNamespace(generic=lambda plan: metadata)
            deadline = types.SimpleNamespace(origin={"budget": dict(process.QUALIFICATION_BUDGET)}, remaining=lambda phase: 1000)
            owner = lambda *args: {"pid": os.getpid(), "uids": [os.getuid()] * 4, "cgroup": "explicit-fixture",
                "command": commands[current_role[0]]}
            with patch.object(profile, "target_process_reader", return_value=process), \
                    patch.object(profile, "direct_process_reader", return_value=facade), \
                    patch.object(profile, "verify_prepared", return_value=None), \
                    patch.object(process, "policy", return_value=None), patch.object(process, "context", side_effect=owner), \
                    patch.object(process, "role_commands", return_value=(commands["waiter"], commands["capture"])), \
                    patch.object(process, "deadline", return_value=deadline) as supplied, \
                    patch.object(process, "load_provider", side_effect=AssertionError("stop before native initialization")) as provider:
                for role, call in (("producer", lambda: profile.run_direct_target_prepared(unit, plan, {})),
                        ("waiter", lambda: profile.run_direct_owned_waiter(unit)),
                        ("capture", lambda: profile.capture_direct_owned_child(unit))):
                    current_role[0] = role
                    with self.assertRaisesRegex(AssertionError, "stop before native"):call()
                self.assertEqual(supplied.call_count, 3)
                self.assertEqual(provider.call_count, 3)

    def test_controller_release_checks_only_the_actual_producer(self):
        """Generic descendant verification skips the producer PID barrier; producer entry retains it."""
        profile = load("profile_selected")
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary);unit = run / "data/explicit-fixture";unit.mkdir(parents=True)
            plan = {"campaign_run_dir": str(run), "unit_id": unit.name, "python": "/explicit-fixture/bin/python"}
            path = unit / "measurement_plan.json";path.write_text(json.dumps(plan))
            (unit / "measurement_prepared.json").write_text(json.dumps({"plan_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}))
            qualifier = types.SimpleNamespace(authorize=lambda plan: None,
                wait_controller_release=lambda *args: (_ for _ in ()).throw(AssertionError("actual producer barrier")))
            with patch.object(profile, "target_qualification_reader", return_value=qualifier):
                with self.assertRaisesRegex(ValueError, "staged profile entrypoint"):profile.verify_prepared(unit, plan)
                with patch.object(profile, "verify_prepared", return_value=None):
                    with self.assertRaisesRegex(AssertionError, "actual producer barrier"):profile.run_prepared(unit)

    def test_new_aa_saved_reader_retains_every_original_raw_check(self):
        """Compare exact old source/ancestry/release/wait/result conditions without its legacy-native dispatch."""
        def function(path, name):return next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == name)
        old = function(SCRIPTS / "nsight_process_identity.py", "validate_saved")
        new = function(SCRIPTS / "profile_target_qualification.py", "validate_configured_capture_records")
        names = {"validate_saved_intent", "read_json", "receipt", "commands", "nsight_command", "validate_snapshot",
            "validate_observation", "stable", "validate_root_wait", "PROTOCOL"}
        class OriginalNames(ast.NodeTransformer):
            """Normalize only explicit metadata attribute qualification for source comparison."""
            def visit_Attribute(self, node):
                if isinstance(node.value, ast.Name) and node.value.id == "metadata" and node.attr in names:
                    return ast.copy_location(ast.Name(id=node.attr, ctx=node.ctx), node)
                return self.generic_visit(node)
        actual = [OriginalNames().visit(copy.deepcopy(n)) for n in new.body[3:-4]]
        self.assertEqual(ast.dump(ast.Module(body=old.body[1:-2], type_ignores=[]), include_attributes=False),
            ast.dump(ast.Module(body=actual, type_ignores=[]), include_attributes=False))
        self.assertEqual(ast.dump(old.body[-1], include_attributes=False),
            ast.dump(OriginalNames().visit(copy.deepcopy(new.body[-1])), include_attributes=False))
        calls = [n for n in ast.walk(new) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
        self.assertEqual(sum(n.func.attr == "validate_saved" and isinstance(n.func.value, ast.Name)
            and n.func.value.id == "reader" for n in calls), 1)

    def test_new_h_backend_hooks_reuse_dedicated_begin_final_and_leave_ordinary_unchanged(self):
        """Run real new backend validators with synthetic image/context premises and no libraries."""
        q, direct, prerequisites = load("profile_target_qualification"), load("direct_measurement"), load("direct_profile_prerequisites")
        fixtures = load("test_profile_qualification_backend", tests=True)
        fixture = fixtures.ProfileQualificationBackendTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
        backend = fixtures.backend
        fixture.plan.pop("configured_isolation")
        fixture.plan.update(hardware_class="h100", direct_binding={"synthetic_fixture": True}, gpu_uuid=backend.CARD_ONE)
        fixture.process["hostname"] = "serval02";fixture.save_plan_and_binding()
        fixture.begin();unused, loaded, work = fixture.complete()
        captured = {"capture_completed": True, "resolved_device_binding": fixture.binding,
            "loaded_gpu_libraries": loaded, "execution_identity": work["execution_identity"]}
        fixture.write(fixture.unit / "capture_result.json", captured)
        fixture.write(fixture.unit / "profile_target_qualification_work.json", work)
        with patch.object(q, "contract", return_value={}), patch.object(q, "adjacent", return_value=backend), \
                patch.object(direct, "adjacent", return_value=q):
            self.assertIs(direct.backend_support(fixture.plan), backend)
            with fixture.live():self.assertEqual(q.verify_bound_backend_begin(fixture.unit, fixture.plan, fixture.binding)["kind"], "profile_qualification_backend_begin")
            proof = direct.capture_backend_completion(fixture.unit, fixture.plan, captured)
            self.assertEqual(proof["profile_target_backend_completion"], q.reference(fixture.unit / backend.FINAL))
            code = fixture.unit / "code";code.mkdir()
            source = code / "direct_measurement.py";source.write_text("# Explicit metadata fixture; never executed.\n")
            staged = types.SimpleNamespace(__file__=str(source), capture_backend_completion=direct.capture_backend_completion)
            with patch.object(prerequisites, "compute_context", return_value=None), \
                    patch.object(prerequisites, "contract", return_value=None), \
                    patch.object(prerequisites.importlib, "import_module", return_value=staged):
                self.assertEqual(prerequisites.complete_capture(fixture.unit, fixture.plan), proof)
            (fixture.unit / prerequisites.PROOF_NAME).write_text("{}")
            with patch.object(prerequisites, "compute_context", return_value=None), \
                    patch.object(prerequisites, "validate_metadata", return_value=None), \
                    patch.object(prerequisites, "target_qualification", return_value=q), \
                    patch.object(prerequisites, "checked_reference", side_effect=AssertionError("baseline proof is still required")):
                with self.assertRaisesRegex(AssertionError, "baseline proof is still required"):
                    prerequisites.completed_capture_binding(fixture.unit, fixture.plan, captured)
        self.assertFalse((fixture.unit / "direct_backend_capture.json").exists())
        self.assertNotIn("direct_backend_image", fixture.plan)
        original_backend = object()
        with patch.object(direct, "backend_support", return_value=types.SimpleNamespace(capture_complete=lambda *args: original_backend)):
            self.assertIs(direct.capture_backend_completion(fixture.unit, {"kind": "selected_configuration_profile"}, captured), original_backend)

    def test_new_h_binding_preparation_never_declares_old_backend_protocol(self):
        """Use real binding metadata code then real new backend preparation with synthetic public files."""
        prerequisites = load("direct_profile_prerequisites")
        fixtures = load("test_profile_qualification_backend", tests=True)
        fixture = fixtures.ProfileQualificationBackendTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
        backend = fixtures.backend
        plan, unit = fixture.plan, fixture.unit
        plan.pop("configured_isolation")
        plan.update(hardware_class="h100", direct_binding={"synthetic_fixture": True}, gpu_uuid=backend.CARD_ONE)
        plan.pop("profile_qualification_backend")
        (unit / backend.SIDECAR).unlink()
        (unit / "direct_history").mkdir();history = unit / "direct_history/inputs.json";history.write_text("{}")
        identity = types.SimpleNamespace(CAPABILITY="explicit-fixture", require_profile=lambda value: None,
            pending_identity=lambda: {}, validate_plan=lambda *args: None)
        support = types.SimpleNamespace(core=lambda: identity,
            adjacent=lambda *args: (_ for _ in ()).throw(AssertionError("old backend must not configure a new qualifier")))
        with patch.object(prerequisites.importlib, "import_module", return_value=support):
            prerequisites.configure_binding(plan, {"explicit-fixture": True}, history, None)
        self.assertIn("direct_binding", plan)
        self.assertNotIn("direct_backend_image", plan)
        backend.configure_plan(plan, fixture.external)
        self.assertEqual(backend.expected_for_plan(plan), fixture.expected)
        self.assertEqual(plan["profile_qualification_backend"]["expected"]["path"], str(unit / backend.SIDECAR))
        invalid = copy.deepcopy(plan);invalid["direct_backend_image"] = {}
        with self.assertRaisesRegex(ValueError, "inherit another protocol"):backend.unit_directory(invalid)


if __name__ == "__main__":
    unittest.main()
