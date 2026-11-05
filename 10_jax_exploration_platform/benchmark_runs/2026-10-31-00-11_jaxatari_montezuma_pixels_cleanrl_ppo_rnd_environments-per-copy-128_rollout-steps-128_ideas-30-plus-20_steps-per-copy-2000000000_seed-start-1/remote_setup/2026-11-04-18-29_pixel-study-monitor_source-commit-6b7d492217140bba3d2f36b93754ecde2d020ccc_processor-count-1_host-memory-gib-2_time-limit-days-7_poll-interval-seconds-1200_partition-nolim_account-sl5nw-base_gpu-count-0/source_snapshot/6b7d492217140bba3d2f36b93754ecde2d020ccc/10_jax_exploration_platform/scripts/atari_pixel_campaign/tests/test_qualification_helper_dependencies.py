"""Check source-sensitive qualification dependencies without arrays, GPUs or live campaign records."""

import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
import advance_adaptive_research as adaptive
import launch_final_training as final
import measure_gpu_scaling as scaling
import measure_selected as measurement
import post_comparison_validation as validation
import profile_selected as profiling
import train_deployed as deployed
import validate_training_deployment as deployment

PREFIX = deployment.PREFIX
QUALIFICATION = {PREFIX + name for name in ("h100_nvl_slurm_qualification.py", "h100_nvl_allocation_observation.py")}
DECLARATION = "H100_NVL_SLURM_QUALIFICATION_MODULE = 'h100_nvl_slurm_qualification.py'\n"


class QualificationHelperTests(unittest.TestCase):
    """Keep legacy receipts unchanged and require both adjacent files for explicitly declared new callers."""

    def verified_callers(self, marker=None, caller=True, target="h100_nvl_slurm_queue.py"):
        """Construct only the already verified source-byte inputs consumed by the declaration rule."""
        return {PREFIX + "post_comparison_validation.py": (
            b"H100_NVL_SLURM_QUEUE_MODULE = 'h100_nvl_slurm_queue.py'\n" if caller else b"# legacy caller\n"),
            PREFIX + "h100_nvl_slurm_queue.py": b"# original queue\n",
            PREFIX + "h100_nvl_slurm_evidence.py": b"# original evidence\n",
            **({PREFIX + target: marker.encode()} if marker is not None else {})}

    def test_legacy_callers_do_not_gain_new_helper_requirements(self):
        """Adding future files to minimum sets does not change old source identities."""
        original = {PREFIX + "train_optimized.py"}
        for verified in (self.verified_callers(), self.verified_callers(DECLARATION, caller=False)):
            before = dict(verified)
            required = validation.helper_requirements(original | QUALIFICATION, verified)
            self.assertFalse(required & QUALIFICATION)
            self.assertEqual(verified, before)

    def test_only_explicit_top_level_assignments_add_both_dependencies(self):
        """Comments, nested declarations and a different filename cannot enable qualification dependencies."""
        for target in ("h100_nvl_slurm_queue.py", "h100_nvl_slurm_evidence.py"):
            verified = self.verified_callers(DECLARATION, target=target)
            self.assertTrue(QUALIFICATION <= validation.helper_requirements(set(), verified))
        for marker in ("# " + DECLARATION, "def unused():\n    " + DECLARATION,
                       "H100_NVL_SLURM_QUALIFICATION_MODULE = 'another_module.py'\n"):
            with self.subTest(marker=marker):
                self.assertFalse(QUALIFICATION & validation.helper_requirements(QUALIFICATION, self.verified_callers(marker)))

    def test_future_producer_runner_and_admission_lists_cover_the_same_dependencies(self):
        """Profile and scaling inherit selected-measurement staging; deployed placement inherits runner staging."""
        for names in (measurement.HELPERS, profiling.HELPERS, scaling.HELPERS, deployed.HELPERS, final.TRAIN_HELPERS):
            self.assertTrue({Path(name).name for name in QUALIFICATION} <= set(names))
        self.assertTrue(QUALIFICATION <= deployment.REQUIRED_HELPERS)

    def test_original_worker_and_fourth_fifth_attempt_tuples_remain_unchanged(self):
        """Future qualification dependencies must not rewrite the base twelve or historical retry tuples."""
        root = SCRIPTS.parents[2]
        def helpers(contents):
            """Read a literal helper tuple from source without importing its runtime."""
            tree = ast.parse(contents)
            return next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "HELPERS" for target in node.targets))
        for name in ("h100_nvl_slurm_queue.py", "infrastructure_retry.py", "pregather_async_retry.py"):
            original = subprocess.check_output(["git", "-C", str(root), "show",
                "3c8b28402a78535f05f00b23905c9249f5dbdf99:" + PREFIX + name])
            self.assertEqual(helpers((SCRIPTS / name).read_bytes()), helpers(original))
            self.assertFalse({Path(path).name for path in QUALIFICATION} & set(helpers(original)))
        self.assertEqual(len(helpers((SCRIPTS / "h100_nvl_slurm_queue.py").read_bytes())), 12)

    def committed_fixture(self, directory, qualified, included):
        """Create a tiny detached Git receipt whose real bytes establish its helper declarations."""
        names = set(deployment.REQUIRED_HELPERS) - QUALIFICATION
        contents = {name: b"# synthetic committed metadata\n" for name in names}
        contents.update(self.verified_callers(DECLARATION if qualified else None))
        contents.update({name: b"# committed adjacent qualification metadata\n" for name in included})
        for relative, raw in contents.items():
            path = directory / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        for command in (["init", "-q"], ["add", "."], ["-c", "user.name=Helper metadata test", "-c", "user.email=test@example.invalid",
                "-c", "core.hooksPath=/dev/null", "commit", "-qm", "Record helper declarations"], ["checkout", "-q", "--detach"]):
            subprocess.run(["git", "-C", str(directory), *command], check=True, capture_output=True)
        commit = subprocess.check_output(["git", "-C", str(directory), "rev-parse", "HEAD"], text=True).strip()
        return {"source_root": str(directory), "source_commit": commit,
                "files": {relative: hashlib.sha256(raw).hexdigest() for relative, raw in contents.items()}}

    def test_old_committed_receipts_remain_valid_without_either_new_file(self):
        """Deployment and adaptive validation keep the exact old helper receipt bytes."""
        with tempfile.TemporaryDirectory() as temporary:
            receipt = self.committed_fixture(Path(temporary), False, set())
            original = json.dumps(receipt, sort_keys=True)
            deployment.committed_files(receipt, deployment.REQUIRED_HELPERS, validation.Evidence())
            adaptive.committed_files(receipt, validation.Evidence(), deployment.REQUIRED_HELPERS)
            self.assertEqual(json.dumps(receipt, sort_keys=True), original)

    def test_new_committed_callers_reject_either_missing_adjacent_file(self):
        """Each missing dependency fails after its pinned caller bytes have actually been verified."""
        for missing in sorted(QUALIFICATION):
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as temporary:
                receipt = self.committed_fixture(Path(temporary), True, QUALIFICATION - {missing})
                with self.assertRaisesRegex(ValueError, "required source helper"):
                    deployment.committed_files(receipt, deployment.REQUIRED_HELPERS, validation.Evidence())
                with self.assertRaisesRegex(ValueError, "required files"):
                    adaptive.committed_files(receipt, validation.Evidence(), deployment.REQUIRED_HELPERS)

    def test_complete_new_receipt_passes_both_consumers(self):
        """Both original-byte validators accept the full newly declared helper set."""
        with tempfile.TemporaryDirectory() as temporary:
            receipt = self.committed_fixture(Path(temporary), True, QUALIFICATION)
            deployment.committed_files(receipt, deployment.REQUIRED_HELPERS, validation.Evidence())
            adaptive.committed_files(receipt, validation.Evidence(), deployment.REQUIRED_HELPERS)

    def test_working_tree_declaration_cannot_replace_the_pinned_git_blob(self):
        """A new uncommitted marker is rejected before it can reinterpret an older receipt."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            receipt = self.committed_fixture(root, False, set())
            (root / PREFIX / "h100_nvl_slurm_queue.py").write_text(DECLARATION)
            with self.assertRaisesRegex(ValueError, "unchanged detached"):
                deployment.committed_files(receipt, deployment.REQUIRED_HELPERS, validation.Evidence())
            with self.assertRaisesRegex(ValueError, "committed receipt"):
                adaptive.committed_files(receipt, validation.Evidence(), deployment.REQUIRED_HELPERS, require_worktree=True)

    def test_fresh_alias_staging_loads_adjacent_metadata_without_module_shadowing(self):
        """A fresh child resolves the staged qualifier and observer rather than preloaded same-name modules."""
        root = SCRIPTS.parents[2]
        source_commit = "3c8b28402a78535f05f00b23905c9249f5dbdf99"
        with tempfile.TemporaryDirectory() as temporary:
            stage = Path(temporary)
            for name in measurement.HELPERS:
                raw = subprocess.check_output(["git", "-C", str(root), "show", source_commit + ":" + PREFIX + name])
                (stage / name).write_bytes(raw)
            program = """import importlib.util,json,sys,types
from pathlib import Path
stage=Path(sys.argv[1])
wrong={}
for name in ('h100_nvl_slurm_qualification','h100_nvl_allocation_observation'):
 wrong[name]=types.ModuleType(name);wrong[name].__file__='/different/'+name+'.py';sys.modules[name]=wrong[name]
spec=importlib.util.spec_from_file_location('staged_post_alias',stage/'post_comparison_validation.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
queue=module.h100_nvl_slurm_validation_module()
for name in wrong:
 loaded=queue.load(name)
 assert Path(loaded.__file__).resolve()==stage/(name+'.py')
 assert sys.modules[name] is wrong[name]
for name in sys.modules:
 assert name.split('.')[0] not in {'jax','numpy','torch','jaxtari'}
print(json.dumps({'adjacent_files':2,'numerical_imports':0}))
"""
            result = subprocess.run([sys.executable, "-I", "-B", "-c", program, str(stage)], capture_output=True, text=True, check=True)
            self.assertEqual(json.loads(result.stdout), {"adjacent_files": 2, "numerical_imports": 0})


if __name__ == "__main__":
    unittest.main()
