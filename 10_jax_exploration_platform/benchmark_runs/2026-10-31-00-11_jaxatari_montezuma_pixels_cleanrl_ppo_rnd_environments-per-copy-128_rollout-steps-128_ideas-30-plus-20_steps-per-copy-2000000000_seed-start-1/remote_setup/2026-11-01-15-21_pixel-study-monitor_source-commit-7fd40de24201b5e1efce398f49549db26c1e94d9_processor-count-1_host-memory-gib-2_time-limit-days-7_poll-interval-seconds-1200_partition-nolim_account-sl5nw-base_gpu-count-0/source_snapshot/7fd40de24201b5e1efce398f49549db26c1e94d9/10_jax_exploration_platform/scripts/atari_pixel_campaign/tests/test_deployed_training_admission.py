"""Check deployed-training admission with synthetic source/evidence and no numerical runtime."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
import deployed_training_admission as admission


class DeployedTrainingAdmissionTests(unittest.TestCase):
    """Admit PCIe-selected counts through their own deployment proof, preserving logical identity."""

    def setUp(self):
        """Reuse complete synthetic global/target fixtures and add one explicit operational specification."""
        specification = importlib.util.spec_from_file_location("operational_deployment_fixture", SCRIPTS / "tests/test_validate_training_deployment.py")
        fixtures = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(fixtures)
        self.fixture = fixtures.TrainingDeploymentTests("test_actual_pcie_evidence_selects_its_own_copy_count_and_backend")
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.setUp()
        self.directory = self.fixture.directory
        self.deployment_specification = self.fixture.write(self.directory / "deployment_specification.json", self.fixture.specification)
        self.deployment = self.fixture.read_historical_deployment(self.fixture.specification, specification_path=self.deployment_specification)
        self.receipt = self.fixture.write(self.directory / "deployment_verification.json", self.deployment)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.operational_assets = Path(temporary.name)
        manifest = self.directory / "operational_metadata/manifest.json"
        manifest.parent.mkdir()
        manifest.write_bytes(Path(self.fixture.target_asset_record["manifest"]).read_bytes())
        self.specification = {"schema_version": 1, "unit_id": self.deployment["unit_id"],
            "source_root": str(self.fixture.source), "source_commit": self.fixture.commit, "hardware_class": "h100",
            "configuration": self.deployment["configuration"], "copy_seeds": self.deployment["copy_seeds"],
            "runtime_environment": self.deployment["runtime_environment"], "final_verification": str(self.fixture.final_path),
            "asset_record": {"root": str(self.operational_assets), "manifest": str(manifest),
                             "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()},
            "jaxtari_source_root": str(self.fixture.public), "deployment_verification": str(self.receipt),
            "deployment_verification_sha256": hashlib.sha256(self.receipt.read_bytes()).hexdigest()}
        self.path = self.fixture.write(self.directory / "training_specification.json", self.specification)

    def write_specification(self):
        """Retain exact test input bytes after an explicit requested-field mutation."""
        self.fixture.write(self.path, self.specification)

    def read_historical_admission(self, path, **options):
        """Read the original operational fixture only through the explicit historical metadata path."""
        return admission.validate_deployed_training_admission(path, historical_read_only=True, **options)

    def test_target_selected_count_is_compatible_with_training_specification_validation(self):
        """The deployed two-policy configuration is admitted separately from the global NVL count of one."""
        original = self.path.read_bytes()
        with patch.object(admission.validation, "validate_training_admission", side_effect=AssertionError("old NVL admission must not be used")), \
                patch.object(Path, "rglob", side_effect=AssertionError("admission must not walk game assets")):
            admitted = self.read_historical_admission(self.path)
        self.assertEqual(admitted["configuration"]["learner"]["n_copies"], 2)
        self.assertEqual(admitted["validation"]["kind"], "deployed_training_admission")
        self.assertEqual(admitted["copy_seeds"], [1001, 1002])
        self.assertEqual(admission.training.validate_specification(admitted), admitted)
        self.assertEqual(admitted["source_root"], self.specification["source_root"])
        self.assertEqual(admitted["asset_record"], self.specification["asset_record"])
        self.assertEqual(admitted["operational_paths"]["source_root"], str(self.fixture.source))
        self.assertEqual(admitted["operational_paths"]["asset_root"], str(self.operational_assets))
        self.assertEqual(self.path.read_bytes(), original)
        for name in ("jax", "numpy", "torch"):
            self.assertNotIn(name, sys.modules)

    def test_api_deployment_receipt_with_explicit_none_path_is_also_rebuilt(self):
        """Direct API receipts remain explicit, with no inferred specification-file search."""
        direct = self.fixture.read_historical_deployment(self.fixture.specification)
        self.assertIsNone(direct["specification_path"])
        self.fixture.write(self.receipt, direct)
        self.specification["deployment_verification_sha256"] = hashlib.sha256(self.receipt.read_bytes()).hexdigest()
        self.write_specification()
        self.assertEqual(self.read_historical_admission(self.path)["validation"]["selected_copies"], 2)

    def test_changed_receipt_hash_or_missing_original_specification_path_is_rejected(self):
        """A receipt cannot be replaced or admitted through an unissued prototype format."""
        self.specification["deployment_verification_sha256"] = "a" * 64
        self.write_specification()
        with self.assertRaisesRegex(ValueError, "required SHA256"):
            self.read_historical_admission(self.path)
        changed = copy.deepcopy(self.deployment)
        changed.pop("specification_path")
        self.fixture.write(self.receipt, changed)
        self.specification["deployment_verification_sha256"] = hashlib.sha256(self.receipt.read_bytes()).hexdigest()
        self.write_specification()
        with self.assertRaisesRegex(ValueError, "explicit specification_path"):
            self.read_historical_admission(self.path)

    def test_changed_raw_target_evidence_invalidates_the_saved_deployment(self):
        """Even still-parseable raw data cannot reuse a saved raw-evidence deployment receipt."""
        path = Path(self.fixture.specification["confirmation_record"])
        path.write_bytes(path.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "deployment raw evidence changed"):
            self.read_historical_admission(self.path)

    def test_operational_config_seed_source_runtime_and_unit_must_match_target(self):
        """Explicit operational fields cannot replace the count/configuration established on PCIe."""
        original = copy.deepcopy(self.specification)
        mutations = {"copy_seeds": [2001, 2002], "source_commit": "a" * 40, "unit_id": "another_final_unit"}
        changed = copy.deepcopy(original["configuration"])
        changed["learner"]["n_copies"] = 1
        mutations["configuration"] = changed
        mutations["runtime_environment"] = {**original["runtime_environment"], "XLA_FLAGS": "--xla_gpu_exclude_nondeterministic_ops=false"}
        for key, value in mutations.items():
            self.specification = {**original, key: value}
            self.write_specification()
            with self.subTest(field=key), self.assertRaises(ValueError):
                self.read_historical_admission(self.path)
        changed = copy.deepcopy(original["configuration"])
        changed["learner"]["anneal_learning_rate"] = 1
        self.specification = {**original, "configuration": changed}
        self.write_specification()
        with self.assertRaisesRegex(ValueError, "exact target"):
            self.read_historical_admission(self.path)

    def test_an_identical_global_proof_at_another_path_is_not_the_declared_identity(self):
        """The operational specification retains the exact globally verified logical proof path."""
        other = self.directory / "another_global_proof.json"
        other.write_bytes(self.fixture.final_path.read_bytes())
        self.specification["final_verification"] = str(other)
        self.write_specification()
        with self.assertRaisesRegex(ValueError, "another global final-verification"):
            self.read_historical_admission(self.path)

    def test_operational_asset_manifest_must_match_the_deployment_asset_identity(self):
        """A different manifest cannot be introduced by changing its operational SHA alongside it."""
        path = Path(self.specification["asset_record"]["manifest"])
        path.write_bytes(path.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "asset manifest differs"):
            self.read_historical_admission(self.path)
        self.specification["asset_record"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.write_specification()
        with self.assertRaisesRegex(ValueError, "game assets differ"):
            self.read_historical_admission(self.path)

    def test_full_ast_fields_and_actual_detached_git_roots_are_required(self):
        """Unknown/missing fields and an archive below a Git root cannot borrow its identity."""
        specification = copy.deepcopy(self.specification)
        specification["configuration"]["learner"].pop("rollout_unroll")
        with self.assertRaisesRegex(ValueError, "every pinned learner/environment field"):
            admission.full_source_configuration(specification, admission.validation.Evidence())
        archive = self.fixture.public / "archive"
        archive.mkdir()
        self.specification["jaxtari_source_root"] = str(archive)
        self.write_specification()
        with self.assertRaisesRegex(ValueError, "unchanged detached Git source"):
            self.read_historical_admission(self.path)

    def test_portability_keeps_logical_specification_and_returns_physical_paths_separately(self):
        """Deleting every original source/evidence path changes transport locations, not training identities."""
        original = self.read_historical_admission(self.path)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        staging = Path(temporary.name)
        manifest = {"schema_version": 1, "kind": "portable_raw_evidence", "files": [], "git_roots": []}
        actual_roots = {}
        for root, name, commit in ((self.fixture.source, "numerical", self.fixture.commit),
                                  (self.fixture.public, "jaxtari", admission.deployment.JAXTARI_COMMIT)):
            target = staging / "repositories" / name
            subprocess.run(["git", "clone", "-q", "--no-hardlinks", str(root), str(target)], check=True)
            subprocess.run(["git", "-C", str(target), "checkout", "-q", "--detach", commit], check=True)
            manifest["git_roots"].append({"original_root": str(root), "relative_path": "repositories/" + name, "commit": commit})
            actual_roots[name] = str(target)
        for root in (self.directory, self.fixture.fixture.directory):
            for path in sorted(root.rglob("*")):
                if not path.is_file() or ".git" in path.relative_to(root).parts:
                    continue
                contents = path.read_bytes()
                digest = hashlib.sha256(contents).hexdigest()
                target = staging / "files" / digest
                target.parent.mkdir(exist_ok=True)
                target.write_bytes(contents)
                manifest["files"].append({"original_path": str(path), "relative_path": "files/" + digest, "sha256": digest})
        manifest_path = staging / "manifest.json"
        manifest_path.write_text(json.dumps(manifest))
        resolver = admission.portable_evidence.ExactPathResolver(manifest_path)
        shutil.rmtree(self.directory)
        shutil.rmtree(self.fixture.fixture.directory)
        moved = self.read_historical_admission(self.path, resolver=resolver)
        self.assertEqual(moved["validation"], original["validation"])
        for key, value in self.specification.items():
            self.assertEqual(moved[key], value)
        self.assertEqual(moved["operational_paths"]["source_root"], actual_roots["numerical"])
        self.assertEqual(moved["operational_paths"]["jaxtari_source_root"], actual_roots["jaxtari"])
        self.assertEqual(moved["operational_paths"]["asset_root"], str(self.operational_assets))
        self.assertEqual(moved["evidence_resolution"]["manifest_sha256"], hashlib.sha256(manifest_path.read_bytes()).hexdigest())
        self.assertFalse(Path(moved["source_root"]).exists())
        self.assertTrue(Path(moved["operational_paths"]["source_root"]).is_dir())
        for name in ("jax", "numpy", "torch"):
            self.assertNotIn(name, sys.modules)


if __name__ == "__main__":
    unittest.main()
