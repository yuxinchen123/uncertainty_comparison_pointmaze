"""Run actual CPU-only orphan/startup failures and require lock retention through kernel closure."""
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parents[1]


class RealCpuDescendantClosureTests(unittest.TestCase):
    """Use isolated native-process-control fixtures; no Nsight, GPU or standing lock is opened."""
    def regression(self, mode):
        """Run the real source cleanup and retain exact temporary process/lock evidence."""
        with tempfile.TemporaryDirectory() as temporary:
            result = subprocess.run(["/usr/bin/python3", "-I", "-S", "-B",
                str(SCRIPTS / "tests/direct_tool_descendants_cpu_worker.py"), temporary, mode],
                capture_output=True, timeout=25, check=False, close_fds=True, pass_fds=())
            self.assertEqual(result.returncode, 0, result.stderr.decode() + result.stdout.decode())
            record = json.loads(result.stdout)
            preserved = os.environ.get("DIRECT_CPU_SCOPE_REVIEW_OUTPUT")
            if preserved:
                target = Path(preserved) / mode
                target.mkdir(parents=True, exist_ok=False)
                mappings = []
                source_root = Path(temporary)
                for source in sorted(source_root.rglob("*.json")):
                    if "code" in source.parts:
                        continue
                    destination = target / source.relative_to(source_root)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source, destination)
                    mappings.append({"original_path": str(source), "preserved_path": str(destination)})
                (target / "worker_stdout.raw").write_bytes(result.stdout)
                (target / "worker_stderr.raw").write_bytes(result.stderr)
                (target / "temporary_evidence_paths.json").write_text(json.dumps(mappings, indent=2) + "\n")
            self.assertTrue(record["passed"])
            self.assertTrue(record["payload_exit_observed_by_independent_pidfd"])
            self.assertTrue(record["temporary_lock_retained_until_all_descendants_exited"])
            self.assertTrue(record["temporary_lock_reacquired_only_after_verified_descendant_exit"])
            self.assertTrue(record["kernel_terminal"]["actual_echild"])
            self.assertFalse(record["standing_gpu_lock_opened"])
            self.assertFalse(record["nsight_executed"])
            self.assertEqual(record["gpu_operations"], 0)
            return record

    def test_startup_timeout_cleans_actual_owned_payload_before_temporary_lock_release(self):
        """Reproduce the rejected startup path with a real different-session CPU payload."""
        record = self.regression("startup_timeout")
        self.assertTrue(record["actual_root_waits"])

    def test_fast_failed_root_and_orphaned_payload_are_reaped_before_lock_release(self):
        """A failed root and double-forked orphan cannot escape the producer's subreaper scope."""
        record = self.regression("orphan_root_failure")
        self.assertIn("middle_pid", record["actual_processes"])


if __name__ == "__main__":
    unittest.main()
