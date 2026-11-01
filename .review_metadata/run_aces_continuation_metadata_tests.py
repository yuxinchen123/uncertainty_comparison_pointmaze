"""Run only the new transport's standard-library fixtures on the exact owned CPU allocation."""

import builtins
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import unittest
from unittest.mock import patch


HERE = Path(__file__).resolve().parent


def main():
    """Pin all four files and block network, numerical imports and every non-Git process."""
    if os.environ.get("SLURM_JOB_ID") != "6781087":
        raise ValueError("these metadata checks require only the currently owned CPU allocation6781087")
    paths = [HERE / name for name in ("continue_aces_autodetect_native_preparation.py",
        "test_aces_autodetect_incremental_native_preparation.py", "aces_autodetect_incremental_native_preparation_driver_plan.json",
        "run_aces_continuation_metadata_tests.py")]
    before = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    original_process, original_import = subprocess.Popen, builtins.__import__
    calls = []

    def local_git(command, *arguments, **options):
        """Allow only bounded local Git for the tiny incremental-bundle fixture."""
        if not isinstance(command, list) or command[0] != "git" or any(value in command for value in ("fetch", "pull", "push", "clone", "ls-remote")):
            raise AssertionError("SSH, installs, native commands and all nonlocal processes are forbidden")
        allowed = {"init", "add", "commit", "rev-parse", "update-ref", "merge-base", "bundle", "show-ref"}
        if not any(value in allowed for value in command):
            raise AssertionError("an unsupported local Git operation was requested")
        if "-C" in command and not Path(command[command.index("-C") + 1]).is_relative_to(Path("/tmp")):
            raise AssertionError("tests may not mutate a real project Git checkout")
        calls.append(command)
        return original_process(command, *arguments, **options)

    def metadata_import(name, *arguments, **options):
        """Stop numerical initialization before executing any package source."""
        if name.split(".")[0] in {"jax", "jaxlib", "numpy", "torch", "scipy", "flax", "optax", "jaxtari"}:
            raise AssertionError("numerical imports are forbidden in metadata checks")
        return original_import(name, *arguments, **options)

    def no_network(*arguments, **options):
        """Reject all actual network and shell operations in this isolated metadata suite."""
        raise AssertionError("network and shell operations are forbidden")

    sys.path.insert(0, str(HERE))
    output, started = io.StringIO(), time.monotonic()
    with patch.object(subprocess, "Popen", side_effect=local_git), patch.object(builtins, "__import__", side_effect=metadata_import), \
            patch.object(socket, "create_connection", side_effect=no_network), patch.object(socket.socket, "connect", side_effect=no_network), \
            patch.object(os, "system", side_effect=no_network):
        suite = unittest.defaultTestLoader.loadTestsFromName("test_aces_autodetect_incremental_native_preparation")
        result = unittest.TextTestRunner(stream=output, verbosity=2).run(suite)
    after = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    if before != after:
        raise ValueError("source changed during the exact-hash test run")
    record = {"schema_version": 1, "kind": "aces_autodetect_incremental_native_preparation_author_tests",
        "at_utc": datetime.now(timezone.utc).isoformat(), "passed": result.wasSuccessful(), "tests": result.testsRun,
        "seconds": time.monotonic() - started, "files": before, "source_unchanged": True,
        "actual_owned_job_id": os.environ["SLURM_JOB_ID"], "actual_owned_step_id": os.environ.get("SLURM_STEP_ID"),
        "process": {"hostname": socket.gethostname(), "pid": os.getpid(),
            "process_start_time": Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19],
            "cgroup": Path("/proc/self/cgroup").read_text()}, "local_git_processes": len(calls),
        "ssh_calls": 0, "network_calls": 0, "installs": 0, "numerical_imports": [], "job_submissions": 0,
        "test_output": output.getvalue(), "workers_permitted": False, "scientific_hypothesis_credit": 0}
    path = HERE / ("aces_continuation_metadata_author_" + datetime.now(timezone.utc).strftime("%Y-%m-%d-%H-%M-%S") + ".json")
    raw = (json.dumps(record, indent=2) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"passed": record["passed"], "tests": record["tests"], "step": record["actual_owned_step_id"],
        "receipt": str(path), "sha256": hashlib.sha256(raw).hexdigest()}))
    if not result.wasSuccessful():
        print(output.getvalue())
        raise SystemExit(1)


if __name__ == "__main__":
    main()
