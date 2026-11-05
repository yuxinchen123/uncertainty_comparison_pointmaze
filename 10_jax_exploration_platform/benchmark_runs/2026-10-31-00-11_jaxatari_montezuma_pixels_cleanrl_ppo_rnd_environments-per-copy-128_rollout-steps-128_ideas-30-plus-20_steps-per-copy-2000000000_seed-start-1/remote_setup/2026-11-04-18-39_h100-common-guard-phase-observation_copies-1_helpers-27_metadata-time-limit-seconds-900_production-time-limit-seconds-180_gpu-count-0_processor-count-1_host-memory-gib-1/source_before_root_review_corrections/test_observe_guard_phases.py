"""Check that diagnostic timing cannot record foreign functions or unbounded output."""
import importlib.util
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location("phase_diagnostic_source", Path(__file__).with_name("observe_guard_phases.py"))
SOURCE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SOURCE)


class PhaseObservationTests(unittest.TestCase):
    """Exercise exact source filtering and finite traces with synthetic metadata calls."""

    def setUp(self):
        """Keep every test artifact in its own disposable directory."""
        self.directory = tempfile.TemporaryDirectory()
        self.original = SOURCE.PACKET
        SOURCE.PACKET = Path(self.directory.name)
        self.path = SOURCE.PACKET / "declared_validator.py"

    def tearDown(self):
        """Leave profiling and the actual diagnostic packet unchanged."""
        sys.setprofile(None)
        SOURCE.PACKET = self.original
        self.directory.cleanup()

    def observer(self, names=("validate_actual",), events=512, bytes=131072):
        """Build one bounded observer for the exact declared synthetic source."""
        return SOURCE.PhaseObserver({"targets": {str(self.path): list(names)},
            "event_limit": events, "trace_byte_limit": bytes})

    def run_observed(self, observer, text, filename=None):
        """Run synthetic validator code while preserving a real function's source name."""
        namespace = {}
        exec(compile(text, str(filename or self.path), "exec"), namespace)
        sys.setprofile(observer)
        try:
            return namespace["validate_actual"]()
        finally:
            sys.setprofile(None)

    def rows(self):
        """Read only the small synthetic timing trace."""
        return [json.loads(line) for line in (SOURCE.PACKET / "validator_phase_times.jsonl").read_text().splitlines()]

    def test_declared_function_has_only_identity_and_timing_fields(self):
        """A real entry/return pair records no argument or result contents."""
        observer = self.observer()
        result = self.run_observed(observer, "def validate_actual():\n    return 'private result content'\n")
        observer.close(True)
        rows = self.rows()
        self.assertEqual(result, "private result content")
        self.assertEqual([row["event"] for row in rows], ["call", "return"])
        self.assertEqual(rows[0]["call_id"], rows[1]["call_id"])
        self.assertGreaterEqual(rows[1]["duration_seconds"], 0)
        self.assertNotIn("private result content", json.dumps(rows))

    def test_same_function_name_in_foreign_source_is_unobserved(self):
        """A matching name alone cannot join another source file."""
        observer = self.observer()
        self.run_observed(observer, "def validate_actual():\n    return 1\n", SOURCE.PACKET / "foreign.py")
        observer.close(True)
        self.assertEqual(self.rows(), [])

    def test_other_functions_in_declared_source_are_unobserved(self):
        """The observer does not collect a trace of unrelated code."""
        observer = self.observer(names=("other_validator",))
        self.run_observed(observer, "def validate_actual():\n    return 1\n")
        observer.close(True)
        self.assertEqual(self.rows(), [])

    def test_nested_entries_retain_distinct_actual_pairs(self):
        """Nested high-level validators retain separate durations and call identities."""
        observer = self.observer(names=("validate_actual", "verify_saved_measurement"))
        self.run_observed(observer, "def verify_saved_measurement():\n    return 1\ndef validate_actual():\n    return verify_saved_measurement()\n")
        observer.close(True)
        rows = self.rows()
        self.assertEqual([row["call_id"] for row in rows], [1, 2, 2, 1])
        self.assertEqual([row["event"] for row in rows], ["call", "call", "return", "return"])

    def test_event_limit_stops_instead_of_growing_trace(self):
        """A too-small declared event budget fails explicitly."""
        observer = self.observer(events=1)
        with self.assertRaisesRegex(ValueError, "small metadata trace bound"):
            self.run_observed(observer, "def validate_actual():\n    return 1\n")
        observer.close(False)
        self.assertEqual(len(self.rows()), 1)

    def test_byte_limit_stops_before_oversized_write(self):
        """No trace line can exceed the declared total byte bound."""
        observer = self.observer(bytes=10)
        with self.assertRaisesRegex(ValueError, "small metadata trace bound"):
            self.run_observed(observer, "def validate_actual():\n    return 1\n")
        observer.close(False)
        self.assertEqual((SOURCE.PACKET / "validator_phase_times.jsonl").stat().st_size, 0)


if __name__ == "__main__":
    output = io.StringIO()
    result = unittest.TextTestRunner(stream=output, verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(PhaseObservationTests))
    packet = Path(__file__).resolve(strict=True).parent
    raw = output.getvalue().encode()
    with (packet / "phase_observer_metadata_tests_stdout.raw").open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    receipt = {"kind": "actual_named_phase_observer_metadata_tests", "job_id": os.environ["SLURM_JOB_ID"],
        "step_id": os.environ["SLURM_STEP_ID"], "tests_run": result.testsRun,
        "failures": len(result.failures), "errors": len(result.errors), "passed": result.wasSuccessful(),
        "source_sha256": hashlib.sha256(packet.joinpath("observe_guard_phases.py").read_bytes()).hexdigest(),
        "raw_stdout_sha256": hashlib.sha256(raw).hexdigest(), "gpu_or_scientific_actions": 0}
    with (packet / "phase_observer_metadata_tests.json").open("x") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(receipt))
    raise SystemExit(0 if result.wasSuccessful() else 1)
