"""Private metadata fixtures for the once-only transport writer; never execute a transport."""
import contextlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("transport_metadata_under_test", HERE / "prepare_root_launch_transport.py")
producer = importlib.util.module_from_spec(spec); spec.loader.exec_module(producer)


class TransportMetadataTests(unittest.TestCase):
    """Mock publication checks only inside a private fixture; no fixture earns publication or run credit."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="correctness-transport-private-metadata-")
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name) / "campaign"
        self.packet = self.run / "remote_setup/private-unit"
        self.unit = self.run / "data/private-unit"
        self.packet.mkdir(parents=True); self.unit.mkdir(parents=True)
        (self.packet / "root_controller_transport_template.py.txt").write_bytes(
            (HERE / "root_controller_transport_template.py.txt").read_bytes())
        for name in ("controller_source_freeze.json", "packet_metadata_freeze.json"):
            (self.packet / name).write_text("{}")
        (self.unit / "measurement_plan.json").write_text("{}")
        self.plan = {"python": "/p/rlprojects/RND/.venvs/atari_pixels_jax/bin/python"}
        self.command = [self.plan["python"], str(self.unit / "code/measure_selected.py"), "--prepared-unit", str(self.unit)]
        self.synthetic_commit = "e" * 40
        patches = [
            mock.patch.object(producer, "P", self.packet),
            mock.patch.object(producer.controller, "RUN", self.run),
            mock.patch.object(producer.controller, "UNIT", self.unit),
            mock.patch.object(producer.controller, "checked_inputs", return_value=(self.plan, self.command, {})),
            mock.patch.object(producer.controller, "check_publication",
                return_value={"path": str(self.packet / "synthetic_review.json"), "sha256": "a" * 64}),
            mock.patch.object(producer.controller, "check_unattempted"),
            mock.patch.object(producer.controller.subprocess, "Popen", side_effect=AssertionError("no process allowed"))]
        for patch in patches:
            patch.start(); self.addCleanup(patch.stop)

    def run_writer(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return producer.main(["--publication-commit", self.synthetic_commit])

    def test_private_metadata_output_changes_only_the_real_supplied_publication_literal(self):
        self.assertEqual(self.run_writer(), 0)
        source = (self.packet / "root_start_controller_once.py").read_text()
        template = (self.packet / "root_controller_transport_template.py.txt").read_text()
        self.assertEqual(source, template.replace(producer.OLD_PUBLISHED_CONTROLLER, self.synthetic_commit))
        record = producer.controller.document(self.packet / "actual_launch_command_preparation.json")
        self.assertEqual(record["publication"], self.synthetic_commit)
        self.assertEqual(record["producer_command"], self.command)
        self.assertIs(record["transport_executed"], False)
        self.assertIsNone(record["actual_controller_pid"])
        self.assertIsNone(record["actual_numerical_child_pid"])

    def test_failed_publication_check_creates_no_output_metadata(self):
        with mock.patch.object(producer.controller, "check_publication", side_effect=ValueError("not an actual published review")):
            with self.assertRaises(ValueError): self.run_writer()
        self.assertFalse((self.packet / "actual_launch_command_preparation.json").exists())
        self.assertFalse((self.packet / "root_start_controller_once.py").exists())

    def test_changed_template_is_rejected_before_output(self):
        (self.packet / "root_controller_transport_template.py.txt").write_text("changed")
        with self.assertRaises(ValueError): self.run_writer()
        self.assertFalse((self.packet / "actual_launch_command_preparation.json").exists())

    def test_prior_dangling_temporary_cannot_be_reused(self):
        marker = self.packet / "actual_launch_command_preparation.json.prepared"
        marker.symlink_to(self.packet / "missing")
        with self.assertRaises(ValueError): self.run_writer()
        self.assertTrue(marker.is_symlink())
        self.assertFalse((self.packet / "root_start_controller_once.py").exists())

    def test_second_metadata_attempt_preserves_first_complete_bytes(self):
        self.run_writer()
        path = self.packet / "actual_launch_command_preparation.json"
        first = path.read_bytes()
        with self.assertRaises(ValueError): self.run_writer()
        self.assertEqual(path.read_bytes(), first)


if __name__ == "__main__":
    unittest.main()
