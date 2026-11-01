"""Check direct-test identity and release ordering without SSH or numerical imports."""

import json
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import launch_direct


class DirectCheckTests(unittest.TestCase):
    """Keep the authorized card and committed test command explicit."""

    def test_commands_use_only_snapshot_test_paths(self):
        """A short training check invokes its separate, explicitly enabled GPU fixture."""
        root = Path("/frozen/platform")
        command = launch_direct.check_command(root, "training-checks", "/pixel/python")
        self.assertEqual(command[:4], ["/pixel/python", "-m", "pytest", "-q"])
        self.assertEqual(command[-1], str(root / "scripts/atari_pixel_campaign/tests/gpu_train_optimized_restart.py"))
        self.assertEqual(launch_direct.check_command(root, "rendering", "/pixel/python")[-2:],
                         ["--candidate", "all"])
        with self.assertRaises(ValueError):
            launch_direct.check_command(root, "train", "/pixel/python")

    def test_identity_is_recorded_before_release(self):
        """A supervisor has an exact birth counter before the shell can acquire card one."""
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            source = run / "snapshot"
            module = SimpleNamespace(source_snapshot=lambda *args: (source, "a" * 40))
            spec = SimpleNamespace(loader=SimpleNamespace(exec_module=lambda value: None))
            response = subprocess.CompletedProcess([], 0, '{"pid":1234,"start_time":"5678"}\n', '')

            def remote(*args, **kwargs):
                """The launch request must find no release file before the PID is returned."""
                self.assertEqual(list((run / "slurm/direct_jobs").glob("*.released")), [])
                return response

            with patch.object(launch_direct.importlib.util, "spec_from_file_location", return_value=spec), \
                 patch.object(launch_direct.importlib.util, "module_from_spec", return_value=module), \
                 patch.object(launch_direct.subprocess, "run", side_effect=remote):
                launch_direct.launch(Path("/repository"), run, "training-checks", 1)
            records = json.loads((run / "processes.json").read_text())
            release = next((run / "slurm/direct_jobs").glob("*.released"))
            self.assertEqual(json.loads(release.read_text()), records[0])
            self.assertEqual(records[0]["start_time"], "5678")
            script = Path(records[0]["script"]).read_text()
            self.assertLess(script.index("test -r " + str(release)), script.index("flock -w"))
            self.assertIn("/tmp/sl5nw_serval02_card1.lock", script)
            self.assertIn("GPU-fed69954-5a71-dbb2-ee91-544e0eb6b8a3", script)
            self.assertIn("RND_RUN_GPU_COMPONENT_TESTS=1", script)
            self.assertIn("RND_COMPONENT_SOURCE_COMMIT=" + "a" * 40, script)
            self.assertIn("'conf-compute','-i',gpu,'-f'", script)
            progress = json.loads((run / "data" / records[0]["unit"] / "progress.json").read_text())
            self.assertEqual(progress["state"], "waiting_for_standing_lock")


if __name__ == "__main__":
    unittest.main()
