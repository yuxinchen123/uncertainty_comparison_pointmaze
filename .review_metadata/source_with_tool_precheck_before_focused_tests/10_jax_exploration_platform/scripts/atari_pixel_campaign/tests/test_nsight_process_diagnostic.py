"""Check diagnostic identity, resource and receipt gates without running Nsight or a device operation."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("synthetic_nsight_diagnostic", SCRIPTS / "diagnose_nsight_process_tree.py")
D = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(D)


class DiagnosticTests(unittest.TestCase):
    """Use private campaign files and a labelled synthetic allocated job."""
    def setUp(self):
        """Create actual-shaped private receipts and exact owned ledger rows."""
        self.temporary = tempfile.TemporaryDirectory(); self.addCleanup(self.temporary.cleanup)
        self.run = Path(self.temporary.name); self.unit = self.run / "data/private"; self.unit.mkdir(parents=True)
        (self.run / "slurm").mkdir()
        self.plan = {"campaign_run_dir": str(self.run), "packet": str(self.run / "remote_setup/private"),
            "helpers": {"source_commit": "a" * 40}}
        D.identity.publish(self.unit / "measurement_plan.json", self.plan)
        self.job = "123456789"
        self.environment = {"SLURM_JOB_ID": self.job, "SLURM_STEP_ID": "0", "SLURM_CPUS_PER_TASK": "1",
            "SLURM_MEM_PER_NODE": "2048", "SLURM_STEP_GPUS": "1", "CUDA_VISIBLE_DEVICES": "1"}
        self.row = [self.job, D.KIND, "a" * 40, str(self.run / "remote_setup/private/job.sbatch")]
        for filename, content in (("submitted_jobids.txt", self.job), ("submitted_jobids_uva.txt", self.job),
                                  ("submissions.tsv", "\t".join(self.row))):
            (self.run / "slurm" / filename).write_text(content + "\n")
        value = {"job_id": self.job, "plan": D.identity.receipt(self.unit / "measurement_plan.json"), "submission_row": self.row}
        for name in ("submission", "release"):
            value[name] = D.identity.publish(self.unit / (name + ".json"), {
                "kind": "owned_nsight_process_diagnostic_" + name, "job_id": self.job, "returncode": 0,
                "plan_sha256": value["plan"]["sha256"], "data_unit": str(self.unit)})
        self.inputs = value
        D.identity.publish(self.unit / "owned_launch_inputs.json", value)

    def check(self, **changes):
        """Run the real ownership reader with explicit synthetic environment metadata."""
        with patch.dict(os.environ, {**self.environment, **changes}, clear=True), \
                patch.object(D.os, "uname", return_value=types.SimpleNamespace(nodename="jaguar03")):
            return D.owned_launch(self.unit, self.plan)

    def test_exact_released_owned_gpu_job_passes_without_query(self):
        """Reading exact own ledgers and receipts never initializes Nsight, CUDA or a scheduler query."""
        with patch.object(D.subprocess, "run", side_effect=AssertionError("tool or scheduler execution")):
            self.assertEqual(self.check(), self.inputs)

    def test_cpu_monitor_or_multiple_devices_or_wrong_resources_refuse(self):
        """The diagnostic cannot run inside the CPU monitor or a different resource shape."""
        for changes in ({"SLURM_JOB_ID": "6783003"}, {"SLURM_STEP_ID": "batch"}, {"SLURM_CPUS_PER_TASK": "8"},
                        {"SLURM_MEM_PER_NODE": "65536"}, {"SLURM_STEP_GPUS": "0,1"},
                        {"CUDA_VISIBLE_DEVICES": "0,1"}, {"CUDA_VISIBLE_DEVICES": ""}):
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, "numeric allocation"):
                self.check(**changes)

    def test_wrong_node_refuses_without_physical_index_inference(self):
        """Logical selectors are preserved as opaque strings and never used to infer a physical identity."""
        with patch.dict(os.environ, self.environment, clear=True), \
                patch.object(D.os, "uname", return_value=types.SimpleNamespace(nodename="other-node")):
            with self.assertRaisesRegex(ValueError, "numeric allocation"):
                D.owned_launch(self.unit, self.plan)

    def test_duplicate_or_conflicting_owned_ledger_row_refuses(self):
        """A matching row cannot hide another recorded command for the same job."""
        path = self.run / "slurm/submissions.tsv"
        path.write_text("\t".join(self.row) + "\n" + self.job + "\tother-command\n")
        with self.assertRaisesRegex(ValueError, "another command"):
            self.check()

    def test_present_invalid_final_sidecar_does_not_wait_or_retry(self):
        """Only an absent real-release file can trigger the bounded startup wait."""
        path = self.unit / "owned_launch_inputs.json"; path.unlink(); path.symlink_to(self.unit / "missing")
        with patch.object(D.time, "sleep", side_effect=AssertionError("invalid file was retried")), self.assertRaises(FileNotFoundError):
            D.wait_for_owned_launch(self.unit, self.plan)

    def test_absent_final_sidecar_expires_without_any_tool_action(self):
        """An unreleased job never starts the actual tool experiment."""
        (self.unit / "owned_launch_inputs.json").unlink()
        with patch.object(D.time, "monotonic", side_effect=[0, 181]), \
                patch.object(D.subprocess, "run", side_effect=AssertionError("tool execution")):
            with self.assertRaisesRegex(ValueError, "did not arrive"):
                D.wait_for_owned_launch(self.unit, self.plan)

    def test_default_command_requires_explicit_role(self):
        """The no-argument diagnostic cannot create an attempt or query a tool."""
        with patch.object(D, "metadata_guard") as guard, self.assertRaises(SystemExit) as stopped:
            D.main([])
        self.assertEqual(stopped.exception.code, 2); guard.assert_not_called()

    def test_changed_tool_bytes_refuse_before_even_the_version_subprocess(self):
        """Never execute an unverified binary to learn whether its recorded identity changed."""
        tool = self.run / "private-unexecuted-nsys"; tool.write_bytes(b"changed synthetic executable; never run")
        plan = {**self.plan, "profile_tool": {"path": str(tool), "sha256": "0" * 64, "version": "synthetic version"}}
        with patch.object(D, "read_plan", return_value=(self.unit, plan)), \
                patch.object(D, "wait_for_owned_launch", return_value=self.inputs), \
                patch.object(D.subprocess, "run") as version, patch.object(D.subprocess, "Popen") as profile:
            with self.assertRaisesRegex(ValueError, "changed before any tool execution"):
                D.run(self.unit)
            version.assert_not_called(); profile.assert_not_called()
        self.assertFalse((self.unit / "tool_version_return.json").exists())
        self.assertFalse((self.unit / "trace").exists())

    def test_tool_hash_is_rechecked_after_the_recorded_version_return(self):
        """A successful simulated version return cannot hide a binary change before the actual profile launch."""
        tool = self.run / "private-unexecuted-nsys"; tool.write_bytes(b"original synthetic executable; never run")
        plan = {**self.plan, "profile_tool": {"path": str(tool), "sha256": D.identity.receipt(tool)["sha256"], "version": "synthetic version"}}
        def version_return(*arguments, **keywords):
            """Simulate a version-only return and a subsequent changed private fixture file."""
            tool.write_bytes(b"changed after simulated version return")
            return types.SimpleNamespace(returncode=0, stdout=b"synthetic version\n", stderr=b"")
        with patch.object(D, "read_plan", return_value=(self.unit, plan)), \
                patch.object(D, "wait_for_owned_launch", return_value=self.inputs), \
                patch.object(D.subprocess, "run", side_effect=version_return) as version, \
                patch.object(D.subprocess, "Popen") as profile:
            with self.assertRaisesRegex(ValueError, "installed Nsight tool changed"):
                D.run(self.unit)
            version.assert_called_once_with([str(tool), "--version"], capture_output=True, timeout=15)
            profile.assert_not_called()
        self.assertTrue((self.unit / "tool_version_return.json").exists())
        self.assertFalse((self.unit / "trace").exists())


if __name__ == "__main__": unittest.main()
