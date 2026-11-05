"""Run the small diagnostic-launcher metadata checks on the existing owned CPU allocation."""
import argparse
import datetime
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import unittest

PACKET = Path(__file__).resolve().parent


def digest(path):
    """Hash a bounded metadata file without loading a native library."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    """Require the existing CPU step, unchanged reviewed sources and local Git fixture commands."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--expected-freeze", required=True)
    args = parser.parse_args()
    if os.environ.get("SLURM_JOB_ID") != "6821437" or not os.environ.get("SLURM_STEP_ID", "").isdigit():
        raise ValueError("this metadata runner requires an actual numeric step of owned CPU allocation 6821437")
    if os.getuid() != os.geteuid() or os.getuid() != 2618919:
        raise ValueError("the actual campaign owner is required")
    receipt = Path(args.receipt)
    if os.path.lexists(receipt):
        raise ValueError("preserve the previous metadata test receipt")
    freeze_path = PACKET / "monitor_launcher_source_freeze.json"
    if digest(freeze_path) != args.expected_freeze:
        raise ValueError("the actual source freeze changed")
    freeze = json.loads(freeze_path.read_text())
    expected = freeze["files"]
    before = {name: digest(name) for name in expected}
    if before != expected:
        raise ValueError("a frozen metadata source changed before testing")

    def audit(event, values):
        """Permit only local Git subprocesses and refuse numerical, network or device work."""
        if event == "subprocess.Popen":
            command = values[1]
            if not isinstance(command, (list, tuple)) or not command or command[0] != "git":
                raise RuntimeError("only private local Git fixture commands are allowed")
        if event in {"os.system", "os.posix_spawn", "socket.__new__", "ctypes.dlopen"}:
            raise RuntimeError("native, network or library operations are forbidden in metadata tests")
        if event == "import" and str(values[0]).split(".")[0] in {"jax", "jaxlib", "numpy", "torch", "pynvml", "cupy"}:
            raise RuntimeError("numerical imports are forbidden in metadata tests")

    sys.addaudithook(audit)
    specification = importlib.util.spec_from_file_location("nsight_launcher_tests", PACKET / "test_launch_monitor_once.py")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    def identities(value):
        """Flatten test groups without dropping duplicate identities."""
        if isinstance(value, unittest.TestSuite):
            return [name for item in value for name in identities(item)]
        return [value.id()]
    names = identities(suite)
    if len(names) != 10 or len(set(names)) != 10:
        raise ValueError("exactly ten distinct focused metadata tests are required")
    output = io.StringIO()
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    result = unittest.TextTestRunner(stream=output, verbosity=2).run(suite)
    after = {name: digest(name) for name in expected}
    unchanged = before == after and digest(freeze_path) == args.expected_freeze
    value = {
        "kind": "persistent_cpu_monitor_launcher_metadata_tests",
        "schema_version": 1,
        "passed": result.wasSuccessful() and unchanged,
        "source_freeze": {"path": str(freeze_path), "sha256": args.expected_freeze},
        "source_files_unchanged": unchanged,
        "test_ids": names, "tests_run": result.testsRun,
        "failures": len(result.failures), "errors": len(result.errors),
        "job_id": os.environ["SLURM_JOB_ID"], "step_id": os.environ["SLURM_STEP_ID"],
        "process_id": os.getpid(), "uid": os.getuid(),
        "started_at_utc": started,
        "finished_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "test_output": output.getvalue(),
        "all_scheduler_replies_are_synthetic": True,
        "actual_scheduler_actions": 0, "tool_executions": 0, "gpu_operations": 0,
        "scientific_credit": 0
    }
    raw = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    descriptor = os.open(receipt, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    print(output.getvalue(), end="")
    print(json.dumps({"passed": value["passed"], "receipt": str(receipt), "sha256": digest(receipt),
                      "tests_run": result.testsRun, "step_id": value["step_id"]}, sort_keys=True))
    return 0 if value["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

