"""Check exact capacity-diagnosis inputs and submission records without GPU work."""

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
sys.path.insert(0, str(SCRIPTS))
SPECIFICATION = importlib.util.spec_from_file_location("capacity_diagnosis_launcher", SCRIPTS / "launch_capacity_diagnoses.py")
LAUNCHER = importlib.util.module_from_spec(SPECIFICATION)
SPECIFICATION.loader.exec_module(LAUNCHER)


class CapacityDiagnosisTests(unittest.TestCase):
    """Keep original case settings and immediate job-ID recording explicit."""

    def setUp(self):
        """Build exact-shape original-reference and two original-case fixtures."""
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.source = SCRIPTS.parents[2]
        defaults = LAUNCHER.measurement.diagnostic.default_configuration(self.source)
        reference = {name: defaults[name] for name in ("learner", "environment")}
        reference["learner"]["n_copies"] = 2
        reference["execution"] = {"compilation_mode": "whole_iteration", "iterations_per_call": 1,
                                   "report_readout": "none", "report_every_iterations": 10}
        self.reference = json.loads(json.dumps(reference))
        self.reference_path = self.write("completed_reference.json", {"case_completed": True,
            "source_commit": LAUNCHER.NUMERICAL_SOURCE, "hardware_class": "jaguar03_rtx_a4500",
            "copies": 2, "reference": self.reference})
        self.cases = []
        for name, changes in LAUNCHER.DIAGNOSES.items():
            self.cases.append(self.write(name + ".case.json", {"hypothesis": name,
                "source_commit": LAUNCHER.NUMERICAL_SOURCE, "hardware_class": "jaguar03_rtx_a4500",
                "seed_start": 1, "configuration": self.reference["learner"], "reference_environment": None,
                "reference_execution": None, "candidate": {"learner": changes, "environment": {}, "execution": {}}}))
        self.assets = self.directory / "assets"
        (self.assets / "jaxtari").mkdir(parents=True)
        (self.assets / "jaxtari/example.txt").write_text("original asset")
        self.tree = self.directory / "assets_tree.sha256"
        self.tree.write_text(hashlib.sha256(b"original asset").hexdigest() + "  jaxtari/example.txt\n")

    def write(self, name, value):
        """Retain fixture inputs as real independently hashable JSON files."""
        path = self.directory / name
        path.write_text(json.dumps(value))
        return path

    def test_derive_exact_failed_variants_without_changing_reference_or_inputs(self):
        """Only the changes recorded in the original immutable cases appear in candidates."""
        original = {path: path.read_bytes() for path in [self.reference_path, *self.cases]}
        variants = LAUNCHER.derive_configurations(self.reference_path, self.cases)
        self.assertEqual(len(variants), 3)
        self.assertEqual(variants[0]["configuration"], self.reference)
        for variant in variants[1:]:
            expected = copy.deepcopy(self.reference)
            expected["learner"].update(LAUNCHER.DIAGNOSES[variant["name"]])
            self.assertEqual(variant["configuration"], expected)
        self.assertTrue(all(path.read_bytes() == contents for path, contents in original.items()))
        self.assertNotIn("jax", sys.modules)
        self.assertNotIn("numpy", sys.modules)

    def test_changed_cases_or_incomplete_reference_cannot_authorize_new_settings(self):
        """A one-copy reference or another candidate is rejected before submission."""
        reference = json.loads(self.reference_path.read_text())
        reference["copies"] = 1
        self.reference_path.write_text(json.dumps(reference))
        with self.assertRaisesRegex(ValueError, "two-copy"):
            LAUNCHER.derive_configurations(self.reference_path, self.cases)
        reference["copies"] = 2
        self.reference_path.write_text(json.dumps(reference))
        case = json.loads(self.cases[0].read_text())
        case["candidate"]["learner"]["learning_rate"] = .01
        self.cases[0].write_text(json.dumps(case))
        with self.assertRaisesRegex(ValueError, "exact implementation"):
            LAUNCHER.derive_configurations(self.reference_path, self.cases)

    def test_asset_manifest_uses_original_tree_and_lives_outside_assets(self):
        """New JSON metadata never adds files to the pinned asset tree it describes."""
        record_path = LAUNCHER.asset_documents(self.tree, self.assets, self.directory / "metadata")
        record = json.loads(record_path.read_text())
        manifest = Path(record["manifest"])
        self.assertEqual(record["sha256"], hashlib.sha256(manifest.read_bytes()).hexdigest())
        self.assertEqual(json.loads(manifest.read_text())["files"], {
            "jaxtari/example.txt": hashlib.sha256(b"original asset").hexdigest()})
        with self.assertRaisesRegex(ValueError, "outside"):
            LAUNCHER.asset_documents(self.tree, self.assets, self.assets / "metadata")

    def test_three_submissions_use_original_allocator_script_and_record_each_id_immediately(self):
        """Each requested one-card job records its exact ID before the next submission."""
        run = self.directory / "run"
        (run / "slurm/jobs").mkdir(parents=True)
        args = types.SimpleNamespace(run_dir=run, source_root=self.source, repo=self.source,
            helper_root=self.source, reference_record=self.reference_path, failed_case=self.cases,
            assets_tree=self.tree, assets_root=self.assets, jaxtari_source_root=self.source)
        calls = []
        def submit(command, **options):
            """Verify the durable prefix of IDs before admitting another sbatch call."""
            ledger = run / "slurm/submitted_jobids_uva.txt"
            self.assertEqual(ledger.read_text().splitlines() if ledger.exists() else [], [str(8000000 + index) for index in range(len(calls))])
            self.assertIn("--cpus-per-task=8", command)
            self.assertIn("--mem=64G", command)
            self.assertIn("--gres=gpu:nvidia_rtx_a4500:1", command)
            self.assertIn("--reservation=sl5nw_fixture", command)
            script = Path(command[-1]).read_text()
            self.assertIn("source " + str(self.source / "10_jax_exploration_platform/scripts/atari_pixel_campaign/worker_environment.sh"), script)
            self.assertIn("--run-probe", script)
            calls.append(command)
            return subprocess.CompletedProcess(command, 0, str(8000000 + len(calls) - 1) + "\n", "")
        with patch.object(LAUNCHER.measurement.diagnostic, "core_identity"), \
                patch.object(LAUNCHER, "frozen_helper", return_value=(self.source, "b" * 40)), \
                patch.object(LAUNCHER.launch_queue, "current_reservation", return_value="sl5nw_fixture"), \
                patch.object(LAUNCHER.subprocess, "run", side_effect=submit):
            ids = LAUNCHER.submit(args)
        self.assertEqual(ids, ["8000000", "8000001", "8000002"])
        self.assertEqual((run / "slurm/capacity_diagnosis_jobids.txt").read_text().splitlines(), ids)
        self.assertTrue(all("isolated-capacity-diagnosis" in row for row in (run / "slurm/submissions.tsv").read_text().splitlines()))
        self.assertFalse((run / "data").exists())


if __name__ == "__main__":
    unittest.main()
