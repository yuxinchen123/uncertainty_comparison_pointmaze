"""Check every staged validator closure without importing array libraries or running jobs."""

import ast
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))


class RetryValidatorDependencyTests(unittest.TestCase):
    """Bind the lazy retry validator to the same files that are staged and hashed."""

    def test_all_selected_producers_stage_and_pin_the_retry_validator(self):
        """Every inherited selected/profile/scaling/diagnostic closure includes the lazy dependency."""
        for name in ("measure_selected", "profile_selected", "measure_gpu_scaling", "diagnose_correctness_cache", "diagnose_allocator_region", "diagnose_pregather_allocator"):
            with self.subTest(producer=name):
                module = importlib.import_module(name)
                self.assertIn("infrastructure_retry.py", module.HELPERS)
                self.assertIn("pregather_async_retry.py", module.HELPERS)
                self.assertEqual(len(module.HELPERS), len(set(module.HELPERS)))
        self.assertNotIn("jax", sys.modules)
        self.assertNotIn("numpy", sys.modules)

    def test_training_launch_and_runtime_pin_the_identical_complete_helper_list(self):
        """The checkpoint helper identity includes exactly the launcher-admitted runtime closure."""
        launcher = importlib.import_module("launch_final_training")
        tree = ast.parse((SCRIPTS / "train_optimized.py").read_text())
        run = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run")
        lists = [ast.literal_eval(node) for node in ast.walk(run) if isinstance(node, ast.Tuple)
                 and all(isinstance(item, ast.Constant) and isinstance(item.value, str) for item in node.elts)
                 and "post_comparison_validation.py" in [item.value for item in node.elts]]
        self.assertEqual(len(lists), 1)
        self.assertEqual(lists[0], launcher.TRAIN_HELPERS)
        self.assertIn("infrastructure_retry.py", launcher.TRAIN_HELPERS)
        self.assertIn("pregather_async_retry.py", launcher.TRAIN_HELPERS)
        deployment = importlib.import_module("validate_training_deployment")
        self.assertIn(deployment.PREFIX + "infrastructure_retry.py", deployment.REQUIRED_HELPERS)
        self.assertIn(deployment.PREFIX + "pregather_async_retry.py", deployment.REQUIRED_HELPERS)
        deployed = importlib.import_module("train_deployed")
        self.assertIn("pregather_async_retry.py", deployed.HELPERS)
        self.assertEqual(len(deployed.HELPERS), len(set(deployed.HELPERS)))

    def test_adaptive_measurement_and_profile_admission_require_the_dependency(self):
        """Future producer receipts cannot omit the retry module while pinning its caller."""
        tree = ast.parse((SCRIPTS / "advance_adaptive_research.py").read_text())
        for function in ("producer_record", "profile_record"):
            node = next(item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name == function)
            tuples = [ast.literal_eval(item) for item in ast.walk(node) if isinstance(item, ast.Tuple)
                      and all(isinstance(value, ast.Constant) and isinstance(value.value, str) for value in item.elts)
                      and "post_comparison_validation.py" in [value.value for value in item.elts]]
            self.assertEqual(len(tuples), 1)
            self.assertIn("infrastructure_retry.py", tuples[0])
            self.assertIn("pregather_async_retry.py", tuples[0])

    def test_legacy_attempt_four_operational_helper_tuple_is_unchanged(self):
        """Recorded fourth-attempt approvals retain the exact original dependency identity."""
        legacy = importlib.import_module("infrastructure_retry")
        self.assertEqual(legacy.HELPERS, ("infrastructure_retry.py", "infrastructure_retry_worker.py", "run_infrastructure_retry.py",
            "controller.py", "queue_worker.py", "ideas.py", "train_optimized.py", "diagnose_gpu_execution.py",
            "post_comparison_validation.py", "measure_gpu_scaling.py", "measure_selected.py"))

    def test_committed_callers_require_new_helper_only_after_they_dispatch_it(self):
        """Old pinned metadata stays valid; new pinned callers cannot omit their new adjacent dependency."""
        adaptive = importlib.import_module("advance_adaptive_research")
        deployment = importlib.import_module("validate_training_deployment")
        validator = importlib.import_module("post_comparison_validation")
        prefix = deployment.PREFIX
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("post_comparison_validation.py", "train_optimized.py"):
                path = root / prefix / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("# old committed metadata helper\n")
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            def committed():
                """Create detached synthetic history without importing its source files."""
                subprocess.run(["git", "-C", str(root), "add", "."], check=True)
                subprocess.run(["git", "-C", str(root), "-c", "user.name=Metadata dependency test",
                    "-c", "user.email=test@example.invalid", "-c", "core.hooksPath=/dev/null", "commit", "-qm", "Record metadata helper source"], check=True)
                commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
                subprocess.run(["git", "-C", str(root), "checkout", "-q", "--detach", commit], check=True)
                return commit
            def source_receipt(commit, names):
                """Hash actual committed caller bytes in the test receipt."""
                return {"source_root": str(root), "source_commit": commit,
                    "files": {prefix + name: hashlib.sha256((root / prefix / name).read_bytes()).hexdigest() for name in names}}
            required = {prefix + name for name in ("post_comparison_validation.py", "train_optimized.py", "pregather_async_retry.py")}
            names = ("post_comparison_validation.py", "train_optimized.py")
            old = source_receipt(committed(), names)
            preserved = json.dumps(old, sort_keys=True)
            adaptive.committed_files(old, validator.Evidence(), required)
            deployment.committed_files(old, required, validator.Evidence())
            self.assertEqual(json.dumps(old, sort_keys=True), preserved)
            (root / prefix / "post_comparison_validation.py").write_text("PREGATHER_ASYNC_RETRY_MODULE = 'pregather_async_retry.py'\n")
            current = source_receipt(committed(), names)
            with self.assertRaisesRegex(ValueError, "required"):
                adaptive.committed_files(current, validator.Evidence(), required)
            with self.assertRaisesRegex(ValueError, "required"):
                deployment.committed_files(current, required, validator.Evidence())
            (root / prefix / "pregather_async_retry.py").write_text("# adjacent metadata dependency\n")
            complete = source_receipt(committed(), (*names, "pregather_async_retry.py"))
            adaptive.committed_files(complete, validator.Evidence(), required)
            deployment.committed_files(complete, required, validator.Evidence())
            changed = json.loads(json.dumps(complete))
            changed["files"][prefix + "post_comparison_validation.py"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "committed"):
                adaptive.committed_files(changed, validator.Evidence(), required)
            with self.assertRaisesRegex(ValueError, "committed"):
                deployment.committed_files(changed, required, validator.Evidence())

    def test_shared_retry_validation_has_no_top_level_project_or_numerical_import(self):
        """The dependency can be added to staged metadata helpers without a circular numerical import."""
        tree = ast.parse((SCRIPTS / "infrastructure_retry.py").read_text())
        imports = []
        for node in tree.body:
            if isinstance(node, ast.Import):
                imports.extend(item.name.split(".")[0] for item in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module.split(".")[0])
        self.assertTrue(set(imports) <= sys.stdlib_module_names)
        self.assertNotIn("jax", imports)
        self.assertNotIn("controller", imports)
        self.assertNotIn("ideas", imports)

    def test_attempt_five_alias_load_uses_only_its_adjacent_metadata_closure(self):
        """A fresh isolated process cannot substitute globally loaded helpers for the staged retry source."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("post_comparison_validation.py", "infrastructure_retry.py", "pregather_async_retry.py"):
                (root / name).write_bytes((SCRIPTS / name).read_bytes())
            code = """
import importlib.util,sys,types
from pathlib import Path
root=Path(sys.argv[1])
assert str(root) not in sys.path
original={}
for name in ('controller','ideas','post_comparison_validation','infrastructure_retry','pregather_async_retry'):
    original[name]=types.ModuleType(name);sys.modules[name]=original[name]
spec=importlib.util.spec_from_file_location('isolated_post_alias',root/'post_comparison_validation.py')
post=importlib.util.module_from_spec(spec);spec.loader.exec_module(post)
retry=post.pregather_retry_validation_module()
assert Path(retry.__file__)==root/'pregather_async_retry.py'
assert Path(retry.infra.__file__)==root/'infrastructure_retry.py'
assert all(sys.modules[name] is value for name,value in original.items())
assert not {'jax','numpy','torch'} & set(sys.modules)
(root/'pregather_async_retry.py').unlink()
try: post.pregather_retry_validation_module()
except FileNotFoundError: pass
else: raise AssertionError('missing staged retry used a global module')
print('isolated attempt five dependencies verified')
"""
            result = subprocess.run([sys.executable, "-I", "-c", code, str(root)], cwd="/", capture_output=True, text=True, check=True)
            self.assertEqual(result.stdout.strip(), "isolated attempt five dependencies verified")

    def test_alias_loaded_staged_validator_resolves_only_adjacent_metadata_helpers(self):
        """Alias loading works without helper sys.path entries or replacing frozen module identities."""
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for name in ("post_comparison_validation.py", "infrastructure_retry.py"):
                (directory / name).write_bytes((SCRIPTS / name).read_bytes())
            row = {"iterations": 1, "copies": 2, "environments_per_copy": 128, "rollout_steps": 128,
                "training_transitions": 32768, "seconds": 1., "memory": {"gpu_uuid": "GPU-test", "other_gpu_processes": [],
                    "sampled_process_peak_bytes": 1, "allocator": {"peak_bytes_in_use": 1}}}
            measured = {"measurement_valid": True, "contaminated": False, "copies": 2, "hardware_class": "jaguar03_rtx_a4500",
                "copy_seeds": [1, 2], "correctness_passed": True, "correctness_attempted": True,
                "correctness": {"passed": True, "complete_iteration": {"leaves": 1}, "supplied_inputs": {
                    name: {"leaves": 1} for name in ("preparation", "fixed_batch", "fixed_batch_update")}},
                "rounds": [{"order": ["reference", "candidate"] if i % 2 == 0 else ["candidate", "reference"],
                            "reference": row, "candidate": row} for i in range(7)]}
            diagnosis = {"kind": "allocator_region_history_diagnosis", "completed": True,
                "eligible_for_formal_campaign_admission": False, "hypothesis_count_contribution": 0,
                "source_commit": "d28e6735e79788f618acb0ebfde8609db1f776af", "runtime_environment": {
                    "JAX_ENABLE_X64": "true", "JAX_DEFAULT_MATMUL_PRECISION": "highest",
                    "XLA_FLAGS": "--xla_gpu_exclude_nondeterministic_ops=true"},
                "paired_measurement": measured, "region_preparation": {}, "hardware": {"gpu_uuid": "GPU-test"}, "helpers": {}}
            record = directory / "diagnosis.json"
            record.write_text(json.dumps(diagnosis))
            code = """
import hashlib,importlib.util,json,sys,types
from pathlib import Path
root=Path(sys.argv[1]); record=Path(sys.argv[2])
assert str(root) not in sys.path
frozen_controller=types.ModuleType('controller'); frozen_ideas=types.ModuleType('ideas')
sys.modules['controller']=frozen_controller;sys.modules['ideas']=frozen_ideas
wrong=types.ModuleType('post_comparison_validation')
def reject(*args,**kwargs): raise AssertionError('resolved an unrelated metadata helper')
wrong.correctness=reject;wrong.measured_round=reject
sys.modules['post_comparison_validation']=wrong
wrong_retry=types.ModuleType('infrastructure_retry');wrong_retry.validate_execution=reject
sys.modules['infrastructure_retry']=wrong_retry
spec=importlib.util.spec_from_file_location('alias_loaded_validator',root/'post_comparison_validation.py')
validator=importlib.util.module_from_spec(spec);spec.loader.exec_module(validator)
retry=validator.retry_validation_module()
assert Path(retry.__file__)==root/'infrastructure_retry.py'
retry.DIAGNOSIS_SHA256=hashlib.sha256(record.read_bytes()).hexdigest()
retry.validate_region=lambda *args: None
retry.validate_git_receipt=lambda *args: None
assert retry.validate_diagnosis({'path':str(record),'sha256':retry.DIAGNOSIS_SHA256},retry.LocalEvidence())['completed']
assert sys.modules['controller'] is frozen_controller and sys.modules['ideas'] is frozen_ideas
assert sys.modules['post_comparison_validation'] is wrong and sys.modules['infrastructure_retry'] is wrong_retry
assert 'jax' not in sys.modules and 'numpy' not in sys.modules
(root/'infrastructure_retry.py').unlink()
try: validator.retry_validation_module()
except FileNotFoundError: pass
else: raise AssertionError('missing adjacent dependency used an unrelated module')
print('adjacent aliases verified')
"""
            result = subprocess.run([sys.executable, "-I", "-c", code, str(directory), str(record)],
                capture_output=True, text=True, check=True, cwd="/")
            self.assertEqual(result.stdout.strip(), "adjacent aliases verified")


if __name__ == "__main__":
    unittest.main()
