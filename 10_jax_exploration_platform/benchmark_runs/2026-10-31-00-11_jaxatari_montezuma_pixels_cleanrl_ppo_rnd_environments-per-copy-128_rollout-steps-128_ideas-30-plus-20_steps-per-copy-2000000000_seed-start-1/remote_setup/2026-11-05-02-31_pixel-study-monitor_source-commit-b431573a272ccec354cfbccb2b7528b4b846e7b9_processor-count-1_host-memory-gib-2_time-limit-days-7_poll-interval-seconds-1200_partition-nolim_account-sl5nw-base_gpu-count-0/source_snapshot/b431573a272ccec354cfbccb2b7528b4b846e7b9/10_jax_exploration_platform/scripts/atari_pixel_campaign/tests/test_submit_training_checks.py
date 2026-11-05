"""Check the Slurm GPU-check command and immediate owned-ID recording without submission."""

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import submit_uva


class TrainingCheckSubmissionTests(unittest.TestCase):
    """Keep the physical GPU resolver and frozen source explicit in the generated script."""

    def test_training_check_uses_resolver_and_records_returned_id(self):
        """One typed A4500 allocation runs the two guarded component checks from its snapshot."""
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            commit = "b" * 40
            with patch.object(submit_uva, "source_snapshot", return_value=(Path("/frozen/platform"), commit)), \
                 patch.object(submit_uva.subprocess, "check_output", side_effect=[
                     "NodeName=jaguar03 ReservationName=sl5nw_reserved", "6779999\n"] ) as command:
                identifier = submit_uva.submit(Path("/repository"), run, "training-checks")
            self.assertEqual(identifier, "6779999")
            self.assertEqual((run / "slurm/submitted_jobids_uva.txt").read_text(), "6779999\n")
            submission = command.call_args.args[0]
            self.assertIn("--gres=gpu:nvidia_rtx_a4500:1", submission)
            self.assertIn("--mem=64G", submission)
            script = Path(submission[-1]).read_text()
            self.assertIn("--resolve-slurm-gpu", script)
            self.assertIn('export CUDA_VISIBLE_DEVICES="$gpu_uuid"', script)
            self.assertIn("RND_COMPONENT_SOURCE_COMMIT=" + commit, script)
            self.assertIn("RND_RUN_GPU_COMPONENT_TESTS=1", script)
            self.assertIn("gpu_train_optimized_restart.py", script)
            self.assertLess(script.index("--resolve-slurm-gpu"), script.index("-m pytest"))
            self.assertIn("training-checks", (run / "slurm/submissions.tsv").read_text())

    def test_unexpected_submit_output_is_not_recorded_as_an_id(self):
        """An invalid scheduler response cannot enter the study's owned-ID ledger."""
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            with patch.object(submit_uva, "source_snapshot", return_value=(Path("/frozen/platform"), "b" * 40)), \
                 patch.object(submit_uva.subprocess, "check_output", side_effect=[
                     "ReservationName=(null)", "unexpected response\n"]):
                with self.assertRaises(RuntimeError):
                    submit_uva.submit(Path("/repository"), run, "training-checks")
            self.assertFalse((run / "slurm/submitted_jobids_uva.txt").exists())


if __name__ == "__main__":
    unittest.main()
