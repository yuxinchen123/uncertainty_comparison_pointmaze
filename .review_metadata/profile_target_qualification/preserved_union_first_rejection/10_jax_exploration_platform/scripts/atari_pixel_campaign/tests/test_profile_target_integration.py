"""Check the new metadata-only profile route without tools, GPU operations or actual study units."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1]
ROOT = SCRIPTS.parents[2]
BASE = "548cbcaef4c1fc28f80817ec9ba93f1e6b2fee4e"
PREFIX = "10_jax_exploration_platform/scripts/atari_pixel_campaign/"


def module(name):
    """Load only exact adjacent standard-library source for these declared metadata fixtures."""
    spec = importlib.util.spec_from_file_location("target_integration_" + name, SCRIPTS / (name + ".py"))
    result = importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


class ProfileTargetIntegrationTests(unittest.TestCase):
    """Use synthetic process-policy premises and keep actual scientific admission closed."""
    def setUp(self):
        """Create one owned temporary directory with exact source-byte hashes."""
        temporary = tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        self.run = Path(temporary.name);(self.run / "data/synthetic-qualification").mkdir(parents=True)
        self.directory = self.run / "data/synthetic-qualification"
        self.facade = module("profile_target_direct_process")
        self.release = module("profile_target_release")
        self.plan = {"kind": "full_original_profile_source_target_qualification", "campaign_run_dir": str(self.run),
            "unit_id": self.directory.name, "capture_process_protocol": self.facade.PROTOCOL,
            "gpu_uuid": self.facade.GPU, "hardware_class": "h100", "direct_binding": {"synthetic_premise": True},
            "direct_profile_protocol": "direct-h100-reviewed-common-profile-version-1", "profile_target_process_control": {},
            "python": "/explicit-synthetic-selected/bin/python", "helpers": {"files": {
                PREFIX + p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in SCRIPTS.glob("*.py")}}}

    def test_new_direct_contract_rejects_old_diagnostic_and_missing_policy(self):
        """The new source cannot reinterpret an old diagnostic or an undeclared ordinary profile."""
        with patch.object(self.facade, "adjacent_helper", return_value=types.SimpleNamespace(policy=lambda plan: None, budget=lambda plan: {"schema_version": 1, "work_seconds": 10200, "cleanup_seconds": 600, "total_seconds": 10800})):
            self.assertIs(self.facade.contract(self.plan, self.directory), self.plan)
            for change in ("old-kind", "missing-policy", "other-target", "legacy-native"):
                plan = copy.deepcopy(self.plan)
                if change == "old-kind":plan["kind"] = "direct_nsight_process_descriptor_diagnostic"
                elif change == "missing-policy":plan.pop("profile_target_process_control")
                elif change == "other-target":plan["hardware_class"] = "jaguar03_rtx_a4500"
                else:plan["native_process_control"] = {}
                with self.assertRaises(ValueError):self.facade.contract(plan, self.directory)

    def test_new_release_keeps_real_direct_kind_and_reserved_unit(self):
        """New-kind release support neither changes old readers nor admits foreign directories."""
        self.assertEqual(self.release.unit(self.directory, self.plan), self.directory)
        for name in ("direct_nsight_process_descriptor_diagnostic", "ordinary"):
            plan = copy.deepcopy(self.plan)
            if name == "ordinary":plan.pop("profile_target_process_control")
            else:plan["kind"] = name
            with self.assertRaises(ValueError):self.release.unit(self.directory, plan)
        with self.assertRaises(ValueError):self.release.unit(self.run, self.plan)

    def test_selected_python_scientific_package_flags_remain_original(self):
        """Only the metadata waiter is isolated; the actual scientific child retains package startup."""
        waiter, capture = self.facade.commands(self.directory, self.plan)
        self.assertEqual(waiter[:4], [self.plan["python"], "-I", "-S", "-B"])
        self.assertEqual(capture, [self.plan["python"], str(self.directory / "code/profile_selected.py"),
            "--direct-owned-capture-unit", str(self.directory)])
        self.assertFalse(hasattr(self.facade, "DiagnosticDeadline"))

    def test_original_descriptor_and_release_checks_are_retained(self):
        """Copies of unchanged raw custody checks preserve their exact source syntax trees."""
        def functions(path):
            return {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.FunctionDef)}
        original, current = functions(SCRIPTS / "direct_profile_process.py"), functions(SCRIPTS / "profile_target_direct_process.py")
        for name in ("descriptor_record", "validate_descriptor", "no_standing_descriptor", "lock_fields", "live_capture_custody"):
            self.assertEqual(original[name], current[name], name)
        original, current = functions(SCRIPTS / "direct_capture_release.py"), functions(SCRIPTS / "profile_target_release.py")
        self.assertEqual(set(original), set(current))
        for name in original.keys() - {"unit"}:self.assertEqual(original[name], current[name], name)

    def test_existing_component_helpers_and_numerical_capture_are_unchanged(self):
        """New metadata plumbing cannot alter old component source or the thirteen numerical files."""
        existing = ("configured_native_process.py", "diagnose_configured_native_process.py", "linux_pidfd_process_control.py",
            "native_process_evidence.py", "nsight_process_identity.py", "direct_tool_descendants.py", "direct_profile_process.py")
        paths = [SCRIPTS / name for name in existing]
        paths += list((ROOT / "10_jax_exploration_platform/src/exploration_platform/atari_pixels").glob("*.py"))
        paths += [SCRIPTS / name for name in ("paired_benchmark.py", "ideas.py", "worker_environment.sh")]
        for path in paths:
            relative = str(path.relative_to(ROOT))
            self.assertEqual(path.read_bytes(), subprocess.check_output(["git", "-C", str(ROOT), "show", BASE + ":" + relative]), relative)
        base = subprocess.check_output(["git", "-C", str(ROOT), "show", BASE + ":" + PREFIX + "profile_selected.py"], text=True)
        core = lambda text: ast.dump(next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef)
            and n.name == "_capture_core"), include_attributes=False)
        self.assertEqual(core(base), core((SCRIPTS / "profile_selected.py").read_text()))

    def test_real_helper_unions_are_closed_and_default_routes_refuse(self):
        """Both real hardware spellings stage exact dependencies while ordinary admission stays absent."""
        profile = module("profile_selected")
        direct = profile.required_helpers(direct=True, direct_profile=True, profile_target=True)
        configured = profile.required_helpers(configured=True, owned_processes=True, configured_native=True,
            configured_prerequisites=True, profile_target=True)
        expected = module("direct_profile_prerequisites").helper_roster({"profile_target_qualification": {},
            "helpers": {"files": self.plan["helpers"]["files"]}})
        self.assertEqual({PREFIX + name for name in direct}, expected)
        for roster in (direct, configured):
            for name in ("profile_target_qualification.py", "profile_target_process.py", "profile_target_direct_process.py",
                "profile_target_release.py", "profile_qualification_backend.py"):
                self.assertIn(name, roster)
            self.assertTrue(all((SCRIPTS / name).is_file() for name in roster))
        with self.assertRaises(ValueError):profile.require_reviewed_profile_process_admission()
        with self.assertRaises(ValueError):module("direct_profile_prerequisites").require_admission()
        self.assertFalse({"jax", "jaxlib", "numpy", "torch"} & sys.modules.keys())


if __name__ == "__main__":
    unittest.main()
