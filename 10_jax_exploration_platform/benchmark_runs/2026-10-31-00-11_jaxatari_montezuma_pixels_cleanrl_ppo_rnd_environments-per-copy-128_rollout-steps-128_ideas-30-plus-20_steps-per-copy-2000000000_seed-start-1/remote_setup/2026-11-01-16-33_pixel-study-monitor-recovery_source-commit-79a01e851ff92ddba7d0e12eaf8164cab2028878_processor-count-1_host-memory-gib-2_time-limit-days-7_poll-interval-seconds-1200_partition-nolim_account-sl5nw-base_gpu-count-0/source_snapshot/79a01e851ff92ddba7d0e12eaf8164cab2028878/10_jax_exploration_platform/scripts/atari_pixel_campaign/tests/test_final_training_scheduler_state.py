"""Check current owned queue evidence before final-training continuation, without real job queries."""

from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import launch_final_training as uva
import launch_deployed_training as aces


class FinalTrainingSchedulerStateTests(unittest.TestCase):
    """Current active states override older terminal accounting within exact cluster ownership."""

    def setUp(self):
        """Prepare separate literal UVA and ACES ledgers with one deliberately shared number."""
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.run = Path(self.temporary.name)
        (self.run / "slurm").mkdir()
        (self.run / "slurm/submitted_jobids_uva.txt").write_text("111\n999\n")
        (self.run / "slurm/submitted_jobids_aces.txt").write_text("222\n999\n")

    def result(self, stdout="", returncode=0, stderr=""):
        """Represent one exact raw scheduler command result."""
        return SimpleNamespace(stdout=stdout, returncode=returncode, stderr=stderr)

    def test_uva_requeued_active_state_overrides_preempted_accounting(self):
        """An exact pending requeue cannot authorize another allocation or checkpoint continuation."""
        with patch.object(uva.subprocess, "check_output", return_value="111|PREEMPTED|\n"), \
                patch.object(uva.subprocess, "run", return_value=self.result("111|PENDING\n")) as query:
            self.assertEqual(uva.job_states(self.run, ["111"]), {"111": "PENDING"})
        self.assertEqual(query.call_args.args[0], ["squeue", "-h", "-j", "111", "-o", "%i|%T"])

    def test_aces_current_running_state_overrides_old_timeout_and_retains_raw_proof(self):
        """The ACES accounting result preserves its raw terminal row while returning the actual active state."""
        def query(command, **_):
            """Return an older accounting row and a newer exact current queue row."""
            return self.result("222|TIMEOUT|0:0|2026-11-01T00:00:00|\n" if command[0] == "sacct" else "222|RUNNING\n")
        with patch.object(aces.subprocess, "run", side_effect=query):
            rows, raw = aces.accounting(self.run, ["222"])
        self.assertEqual(rows["222"]["state"], "RUNNING")
        self.assertTrue(rows["222"]["active_in_scheduler"])
        self.assertIn("|TIMEOUT|", raw["stdout"])
        self.assertEqual(raw["current_queue"]["stdout"], "222|RUNNING\n")

    def test_missing_accounting_can_only_yield_a_proven_current_active_state(self):
        """A current exact pending row is usable; two missing records can never mean terminal."""
        with patch.object(uva.subprocess, "check_output", return_value=""), \
                patch.object(uva.subprocess, "run", return_value=self.result("111|PENDING\n")):
            self.assertEqual(uva.job_states(self.run, ["111"]), {"111": "PENDING"})
        with patch.object(uva.subprocess, "check_output", return_value=""), \
                patch.object(uva.subprocess, "run", return_value=self.result()), self.assertRaisesRegex(ValueError, "unknown"):
            uva.job_states(self.run, ["111"])

    def test_foreign_cluster_ids_are_rejected_before_any_queue_or_accounting_query(self):
        """An equal user identity or a foreign ledger entry cannot authorize the other cluster."""
        for operation, identifier in ((uva.job_states, "222"), (aces.accounting, "111")):
            with self.subTest(identifier=identifier), patch.object(uva.subprocess, "run") as queue, \
                    patch.object(uva.subprocess, "check_output") as accounting, self.assertRaises(ValueError):
                operation(self.run, [identifier])
            queue.assert_not_called()
            accounting.assert_not_called()
        with patch.object(uva.subprocess, "run", return_value=self.result("999|PENDING\n")):
            self.assertEqual(uva.active_owned_jobs(self.run, ["999"], "aces")[0], {"999": "PENDING"})

    def test_duplicate_foreign_unknown_or_terminal_queue_rows_are_rejected(self):
        """Malformed current records cannot be ignored in favor of older terminal accounting."""
        for text in ("111|PENDING\n111|RUNNING\n", "222|RUNNING\n", "111|UNRECOGNIZED\n", "111|COMPLETED\n", "111|RUNNING|extra\n"):
            with self.subTest(text=text), patch.object(uva.subprocess, "run", return_value=self.result(text)), \
                    self.assertRaisesRegex(ValueError, "duplicate, foreign or unknown"):
                uva.active_owned_jobs(self.run, ["111"], "uva")

    def test_failed_query_is_not_empty_and_exact_invalid_job_requires_accounting(self):
        """Only Slurm's exact absent-job response can be paired with a known terminal record."""
        with patch.object(uva.subprocess, "check_output", return_value="111|COMPLETED|\n"), \
                patch.object(uva.subprocess, "run", return_value=self.result(returncode=2, stderr="scheduler unavailable")), \
                self.assertRaisesRegex(ValueError, "unproved"):
            uva.job_states(self.run, ["111"])
        with patch.object(uva.subprocess, "check_output", return_value="111|COMPLETED|\n"), \
                patch.object(uva.subprocess, "run", return_value=self.result(returncode=1,
                    stderr="slurm_load_jobs error: Invalid job id specified\n")):
            self.assertEqual(uva.job_states(self.run, ["111"]), {"111": "COMPLETED"})


if __name__ == "__main__":
    unittest.main()
