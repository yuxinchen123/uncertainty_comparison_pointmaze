"""Run the real frozen H100 metadata gate with only the reviewed selector changed in memory."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import unittest

SCRIPTS = Path(__file__).resolve().parents[1]
PREFIX = "10_jax_exploration_platform/scripts/atari_pixel_campaign/"
RUN = Path("/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/"
    "2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_"
    "ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1")
UNIT_NAME = ("2026-11-05-14-01_profile-source-qualification_h100-nvl_serval02-card-1_copies-1_environments-128_"
    "rollout-steps-128_epochs-4_minibatches-4_warmup-rollouts-50_reference-updates-1_seed-start-1_hours-3_helper-source-2aa6df92f7d9")
PLAN_SHA256 = "ed8fb070f45b17170dee4b5274d5d053715fea359bf5109c63fc4ebe24907c59"


class SharedPrerequisiteTests(unittest.TestCase):
    """No plans, recorded source files, helper checks or upstream evidence are rewritten."""
    @classmethod
    def setUpClass(cls):
        """Load actual staged standard-library metadata, then replace only the selector under review."""
        cls.unit = RUN / "data" / UNIT_NAME
        raw = (cls.unit / "measurement_plan.json").read_bytes()
        if hashlib.sha256(raw).hexdigest() != PLAN_SHA256:
            raise AssertionError("the actual frozen H100 plan differs")
        cls.plan = json.loads(raw)
        if str(Path(sys.executable).absolute()) != cls.plan["python"]:
            raise AssertionError("run this metadata fixture with its actual selected Python and -I -S -B")
        spec = importlib.util.spec_from_file_location("actual_h100_staged_profile_metadata", cls.unit / "code/profile_selected.py")
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)
        cls.original = staticmethod(cls.module.configured_profile_reader)
        tree = ast.parse((SCRIPTS / "profile_selected.py").read_bytes())
        selected = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "configured_profile_reader")
        exec(compile(ast.Module(body=[selected], type_ignores=[]), str(SCRIPTS / "profile_selected.py"), "exec"), cls.module.__dict__)
        cls.corrected = staticmethod(cls.module.configured_profile_reader)

    def test_original_failure_then_real_verify_prepared_gate_passes(self):
        """The actual complete nonwriting gate passes after exactly one source-function override."""
        self.module.configured_profile_reader = self.original
        try:
            with self.assertRaisesRegex(ValueError, "prerequisite helper needs its explicit profile protocol"):
                self.module.verify_prepared(self.unit, self.plan)
        finally:
            self.module.configured_profile_reader = self.corrected
        self.assertIsNone(self.module.verify_prepared(self.unit, self.plan))
        self.assertEqual(hashlib.sha256((self.unit / "measurement_plan.json").read_bytes()).hexdigest(), PLAN_SHA256)

    def test_h100_shared_dependency_does_not_select_aa_reader(self):
        """Actual H100 source/target and direct protocol checks precede returning no A4500 reader."""
        self.assertIsNone(self.module.configured_profile_reader(self.plan))
        self.assertIn(PREFIX + "configured_profile_prerequisites.py", self.plan["helpers"]["files"])
        self.assertNotIn("configured_profile_prerequisite_protocol", self.plan)

    def test_missing_direct_target_or_protocol_and_mixed_roles_refuse(self):
        """A staged helper cannot activate missing, mixed or fabricated role declarations."""
        changes = [lambda plan: plan.pop("direct_profile_protocol"),
            lambda plan: plan.update(direct_profile_protocol="another-direct-protocol"),
            lambda plan: plan.pop("profile_target_process_control"),
            lambda plan: plan.pop("profile_target_qualification"),
            lambda plan: plan.update(configured_isolation={}),
            lambda plan: plan.update(configured_native_process_control={}),
            lambda plan: plan.update(native_process_control={}),
            lambda plan: plan.update(configured_profile_prerequisite_protocol={}),
            lambda plan: plan.update(hardware_class="jaguar03_rtx_a4500")]
        for index, change in enumerate(changes):
            plan = copy.deepcopy(self.plan);change(plan)
            with self.subTest(index=index), self.assertRaises(ValueError):
                self.module.configured_profile_reader(plan)

    def test_shared_source_hash_and_ordinary_admission_remain_exact(self):
        """Changed shared bytes and an ordinary profile without actual admission cannot pass."""
        plan = copy.deepcopy(self.plan)
        plan["helpers"]["files"][PREFIX + "configured_profile_prerequisites.py"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "shared H100 prerequisite"):
            self.module.configured_profile_reader(plan)
        plan = copy.deepcopy(self.plan)
        plan.pop("profile_target_qualification");plan["kind"] = "selected_configuration_profile"
        with self.assertRaises(ValueError):self.module.configured_profile_reader(plan)


if __name__ == "__main__":
    unittest.main()
