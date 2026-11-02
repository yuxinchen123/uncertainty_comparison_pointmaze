"""Verify original-reference component scope with standard-library fixtures only."""

import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import unittest
import tempfile
from unittest import mock

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
import run_original_scaling_component as component


class OriginalComponentTests(unittest.TestCase):
    """Keep resource checks separate from optimization selection and final-study evidence."""

    def reference(self):
        """Read literal original defaults through syntax trees, without array imports."""
        root = SCRIPTS.parents[2]
        defaults = component.scaling.diagnostic.default_configuration(root)
        config = {"learner": defaults["learner"], "environment": defaults["environment"],
                  "execution": {"compilation_mode": "whole_iteration", "iterations_per_call": 1,
                                "report_readout": "none", "report_every_iterations": 10}}
        return root, {"source_commit": component.ORIGINAL_SOURCE, "hardware_class": "jaguar03_rtx_a4500",
            "copies": 1, "case_completed": True, "measurement_valid": True, "correctness_passed": True,
            "reference": json.loads(json.dumps(config))}

    def test_only_exact_original_complete_reference_is_accepted(self):
        """An optimized renderer, grouped driver or partial case cannot enter this check."""
        root, record = self.reference()
        self.assertEqual(component.original_configuration(root, record), record["reference"])
        for mutation in ("optimized", "grouped", "incomplete", "wrong_source"):
            changed = copy.deepcopy(record)
            if mutation == "optimized":
                changed["reference"]["environment"]["cache_room_backgrounds"] = True
            elif mutation == "grouped":
                changed["reference"]["execution"]["iterations_per_call"] = 5
            elif mutation == "incomplete":
                changed["case_completed"] = False
            else:
                changed["source_commit"] = "a" * 40
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                component.original_configuration(root, changed)
        self.assertFalse(any(name in sys.modules for name in ("jax", "numpy", "torch")))

    def test_preparation_proves_allocation_and_records_zero_final_credit_before_execution(self):
        """Mock numerical preparation only and retain real component-scope file publication."""
        source, reference = self.reference()
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            (run / "slurm").mkdir()
            (run / "slurm/submitted_jobids_uva.txt").write_text("123\n")
            inputs = {}
            for name, value in (("completed_reference", reference), ("configuration", reference["reference"]),
                                ("runtime_environment", {}), ("asset_record", {})):
                path = run / (name + ".json")
                path.write_text(json.dumps(value))
                inputs[name] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            launch = {"schema_version": 1, "run_dir": str(run), "source_root": str(source),
                "source_commit": component.ORIGINAL_SOURCE, "gpu_count": 8, "unit_id": "synthetic_original_component",
                "inputs": inputs, "python": "/synthetic/venv/python",
                **{name: inputs[name]["path"] for name in ("configuration", "runtime_environment", "asset_record")}}
            path = run / "launch.json"
            path.write_text(json.dumps(launch))
            directory = run / "data" / launch["unit_id"]
            original = Path.read_text
            def read(file, *args, **kwargs):
                """Supply an owned synthetic compute cgroup without touching actual scheduler state."""
                return "0::/slurm/job_123/step_0\n" if file == Path("/proc/self/cgroup") else original(file, *args, **kwargs)
            with mock.patch.dict(os.environ, {"SLURM_JOB_ID": "123", "OMP_NUM_THREADS": "8", "SLURM_CPUS_PER_TASK": "64"}), \
                    mock.patch.object(Path, "read_text", read), mock.patch.object(component.socket, "gethostname", return_value="jaguar03"), \
                    mock.patch.object(component.subprocess, "check_output", return_value="\n".join("GPU-" + str(i) for i in range(8))), \
                    mock.patch.object(component.scaling, "allocation_proof", return_value={"synthetic": "proof"}) as proof, \
                    mock.patch.object(component.scaling, "prepare", return_value=directory) as prepare:
                actual, python = component.prepare_component(path)
                self.assertEqual((actual, python), (directory, "/synthetic/venv/python"))
                self.assertEqual(len(prepare.call_args.args[0].gpu_uuid), 8)
                self.assertEqual(proof.call_args.args[0]["gpu_count"], 8)
                scope = json.loads((directory / "component_scope.json").read_text())
                self.assertFalse(scope["formal_scaling_credit"])
                self.assertEqual((scope["rounds"], scope["updates_per_round"]), (7, 10))
                self.assertEqual(scope["hypothesis_count_contribution"], 0)


if __name__ == "__main__":
    unittest.main()
