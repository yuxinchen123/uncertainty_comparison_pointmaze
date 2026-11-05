"""Check the real CPU observation, durable JSON and unchanged strict reader boundary."""
import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

SCRIPTS = Path(__file__).resolve().parents[1]
PREFIX = "10_jax_exploration_platform/scripts/atari_pixel_campaign/"


def function(name):
    """Read the exact current caller rather than a copied implementation."""
    tree = ast.parse((SCRIPTS / "profile_selected.py").read_bytes())
    return next(value for value in tree.body if isinstance(value, ast.FunctionDef) and value.name == name)


def metadata_reader():
    """Execute the actual source-hash checked loader without importing scientific modules."""
    selected = [function("owned_processes"), function("process_reader")]
    namespace = {"SCRIPTS": SCRIPTS, "importlib": __import__("importlib"),
        "OWNED_PROCESS_PROTOCOL": "nsight-owned-descendant-observation-version-1",
        "training": SimpleNamespace(file_sha256=lambda path: hashlib.sha256(path.read_bytes()).hexdigest())}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(SCRIPTS / "profile_selected.py"), "exec"), namespace)
    plan = {"hardware_class": "jaguar03_rtx_a4500", "configured_isolation": {},
        "capture_process_protocol": namespace["OWNED_PROCESS_PROTOCOL"],
        "helpers": {"files": {PREFIX + "nsight_process_identity.py":
            hashlib.sha256((SCRIPTS / "nsight_process_identity.py").read_bytes()).hexdigest()}}}
    return namespace, plan


class SavedObservationBoundaryTests(unittest.TestCase):
    """Use one blocked own CPU child; saved-data negatives perform no further process work."""
    @classmethod
    def setUpClass(cls):
        """Retain actual process bytes and normal wait after executing the current three-statement boundary."""
        cls.temporary = tempfile.TemporaryDirectory()
        cls.directory = Path(cls.temporary.name)
        namespace, cls.plan = metadata_reader()
        cls.reader = namespace["process_reader"](cls.plan)
        parent = cls.reader.ProcReader().snapshot(os.getpid(), os.getuid())
        command = [sys.executable, "-I", "-S", "-B", "-c",
            "import sys; print('ready', flush=True); sys.stdin.buffer.read(1)"]
        child = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        processes = None
        try:
            # Only this known child is read, pinned and observed while blocked on its own input.
            if child.stdout.readline() != b"ready\n":
                raise AssertionError("the own CPU child did not acknowledge its isolated standard-library startup")
            actual = cls.reader.ProcReader().snapshot(child.pid, os.getuid(), parent["cgroup"])
            processes = cls.reader.OwnedProcesses(child.pid, actual["process_start_time"], parent)
            cls.claim = {"pid": child.pid, "process_start_time": actual["process_start_time"]}
            cls.parent, cls.command = parent, command
            cls.live = processes.observe(child.pid, actual["process_start_time"], command)
            if not all(type(value) is int for value in cls.live["nodes"]):
                raise AssertionError("the actual live collector no longer has the reproduced integer-key boundary")
            try:
                cls.reader.validate_observation(cls.live, cls.claim, command, parent)
            except ValueError as error:
                cls.old_live_failure = str(error)
            else:
                raise AssertionError("the strict reader unexpectedly accepted live integer PID keys")

            # Before: {actual_pid: raw_process}; after durable JSON: {"actual_pid": raw_process}.
            caller = function("capture_configured_native_processes")
            statements = next(value.body for value in ast.walk(caller) if isinstance(value, ast.Try))
            start = next(index for index, value in enumerate(statements) if isinstance(value, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "observation" for target in value.targets))
            boundary = ast.Module(body=statements[start:start + 3], type_ignores=[])
            scope = {"processes": processes, "claim": cls.claim, "intent": {"capture_command": command, "parent": parent},
                "observer": cls.reader, "directory": cls.directory}
            exec(compile(boundary, str(SCRIPTS / "profile_selected.py"), "exec"), scope)
            cls.saved = cls.reader.read_json(cls.directory / "capture_process_observation.json")
            cls.reference = cls.reader.receipt(cls.directory / "capture_process_observation.json")
            cls.rebuilt = cls.reader.validate_observation(cls.saved, cls.claim, command, parent)
        finally:
            # This fixture asks its own blocked child to exit normally and waits on the original Popen object.
            if processes is not None:processes.close()
            child.stdin.write(b"x");child.stdin.flush()
            cls.child_stdout, cls.child_stderr = child.communicate(timeout=10)
            cls.returncode = child.returncode
        destination = os.environ.get("PROFILE_OBSERVATION_TEST_ARTIFACT_DIR")
        if destination:
            output = Path(destination)
            cls.reader.publish(output / "actual_cpu_observation.json", cls.saved)
            cls.reader.publish(output / "actual_cpu_wait.json", {"pid": child.pid, "process_start_time": actual["process_start_time"],
                "returncode": cls.returncode, "stdout": cls.child_stdout.decode(), "stderr": cls.child_stderr.decode(),
                "old_live_reader_failure": cls.old_live_failure, "environment_transitions": 0, "learner_updates": 0,
                "gpu_operations": 0, "scientific_credit": 0})

    @classmethod
    def tearDownClass(cls):
        """Remove only this fixture's temporary local directory."""
        cls.temporary.cleanup()

    def test_actual_live_publish_strict_reader_and_normal_wait(self):
        """The actual live caller validates the exact published JSON and retains its raw identity."""
        self.assertIn("canonical PID strings", self.old_live_failure)
        self.assertTrue(all(type(value) is str for value in self.saved["nodes"]))
        self.assertEqual(self.rebuilt["capture"]["pid"], self.claim["pid"])
        self.assertEqual(self.returncode, 0)
        self.assertEqual(self.child_stderr, b"")
        self.assertEqual(self.reader.receipt(self.directory / "capture_process_observation.json"), self.reference)

    def test_noncanonical_saved_keys_and_changed_raw_hash_refuse(self):
        """Serialization never permits PID aliases or a changed raw command digest."""
        for change in ("key", "raw hash"):
            value = copy.deepcopy(self.saved)
            key = str(self.claim["pid"])
            if change == "key":value["nodes"]["0" + key] = value["nodes"].pop(key)
            else:value["nodes"][key]["cmdline_sha256"] = "0" * 64
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.reader.validate_observation(value, self.claim, self.command, self.parent)
        with self.assertRaisesRegex(ValueError, "existing final or partial"):
            self.reader.publish(self.directory / "capture_process_observation.json", self.saved)

    def test_missing_or_changed_staged_reader_hash_refuses(self):
        """The real loader rejects missing protocol and substituted source metadata before loading a helper."""
        namespace, plan = metadata_reader()
        for change in ("source hash", "protocol"):
            changed = copy.deepcopy(plan)
            if change == "source hash":changed["helpers"]["files"][PREFIX + "nsight_process_identity.py"] = "0" * 64
            else:changed["capture_process_protocol"] = "another-protocol"
            with self.subTest(change=change), self.assertRaises(ValueError):namespace["process_reader"](changed)

    def test_only_the_failed_live_caller_changes(self):
        """The new call matches the existing H100 and prior A4500 saved-JSON boundary."""
        caller = function("capture_configured_native_processes")
        calls = [value for value in ast.walk(caller) if isinstance(value, ast.Call)
            and isinstance(value.func, ast.Attribute) and value.func.attr == "validate_observation"]
        self.assertEqual(len(calls), 1)
        first = calls[0].args[0]
        self.assertIsInstance(first, ast.Call)
        self.assertEqual(ast.unparse(first), "observer.read_json(directory / 'capture_process_observation.json')")


if __name__ == "__main__":
    unittest.main()
