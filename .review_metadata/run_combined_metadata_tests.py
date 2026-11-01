"""Run the combined-source metadata checks only inside the current owned CPU job."""

import argparse
import builtins
import ctypes
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "10_jax_exploration_platform/scripts/atari_pixel_campaign"


def main():
    """Keep exact source hashes, real process identity and complete test output for every attempt."""
    parser = argparse.ArgumentParser()
    parser.add_argument("label")
    parser.add_argument("modules", nargs="+")
    arguments = parser.parse_args()
    if os.environ.get("SLURM_JOB_ID") != "6781087":
        raise ValueError("combined metadata tests require owned CPU job 6781087")
    sys.path[:0] = [str(ROOT / ".review_metadata"), str(SCRIPTS), str(SCRIPTS / "tests")]
    paths = [*SCRIPTS.glob("*.py"), *(SCRIPTS / "tests").glob("*.py"),
             ROOT / ".review_metadata/test_combined_exact_eight_memory.py", Path(__file__)]
    hashes = lambda: {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    before = hashes()
    original_import, original_process = builtins.__import__, subprocess.Popen
    blocked = {"jax", "jaxlib", "numpy", "torch", "cupy", "flax", "optax", "scipy"}
    calls = []

    def standard_library(name, *args, **kwargs):
        """Reject numerical initializers before any package code executes."""
        if name.split(".")[0] in blocked:
            raise AssertionError("numerical import forbidden: " + name)
        return original_import(name, *args, **kwargs)

    def local_git(command, *args, **kwargs):
        """Allow only local source reads and temporary Git fixtures."""
        if not isinstance(command, (list, tuple)) or command[0] != "git" or kwargs.get("shell"):
            raise AssertionError("metadata tests permit only bounded local Git processes")
        allowed = {"init", "add", "commit", "checkout", "clone", "rev-parse", "symbolic-ref", "status",
                   "ls-tree", "show", "diff", "ls-files", "hash-object", "cat-file", "rev-list"}
        operation = next((item for item in command[1:] if item in allowed), None)
        if operation is None or any("://" in str(item) or item in {"fetch", "pull", "push", "remote", "ls-remote"} for item in command):
            raise AssertionError("remote Git and unknown commands are forbidden")
        if operation == "clone" and (not Path(command[-2]).is_absolute() or not Path(command[-2]).exists()
                                      or not Path(command[-1]).is_absolute()):
            raise AssertionError("clones require exact local absolute source and destination paths")
        if operation in {"add", "commit", "checkout"}:
            target = Path(command[command.index("-C") + 1])
            if not target.is_relative_to(Path(tempfile.gettempdir())):
                raise AssertionError("a metadata test cannot mutate a real project checkout")
        calls.append(operation)
        return original_process(command, *args, **kwargs)

    def forbidden(*args, **kwargs):
        """Keep network, shell and actual shared-library loading outside the test process."""
        raise AssertionError("network, shell and actual shared-library calls are forbidden")

    output, started = io.StringIO(), time.monotonic()
    with patch.object(builtins, "__import__", side_effect=standard_library), \
            patch.object(subprocess, "Popen", side_effect=local_git), \
            patch.object(socket, "create_connection", side_effect=forbidden), \
            patch.object(socket.socket, "connect", side_effect=forbidden), \
            patch.object(ctypes, "CDLL", side_effect=forbidden), patch.object(os, "system", side_effect=forbidden):
        suite = unittest.defaultTestLoader.loadTestsFromNames(arguments.modules)
        result = unittest.TextTestRunner(stream=output, verbosity=2).run(suite)
    after = hashes()
    if before != after or blocked.intersection(sys.modules):
        raise AssertionError("source changed or a numerical module was imported")
    record = {"schema_version": 1, "kind": "combined_final_memory_metadata_verification",
        "at_utc": datetime.now(timezone.utc).isoformat(), "label": arguments.label,
        "job_id": os.environ["SLURM_JOB_ID"], "step_id": os.environ.get("SLURM_STEP_ID"),
        "process": {"hostname": socket.gethostname(), "pid": os.getpid(),
            "process_start_time": Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19],
            "cgroup": Path("/proc/self/cgroup").read_text()},
        "source_hashes_before": before, "source_hashes_after": after, "modules": arguments.modules,
        "tests": result.testsRun, "passed": result.wasSuccessful(), "seconds": time.monotonic() - started,
        "test_output": output.getvalue(), "local_git_calls": len(calls), "numerical_imports": False,
        "real_library_loads": 0, "network_calls": 0, "live_scheduler_queries": 0, "ssh_connections": 0,
        "gpu_jobs": 0, "jobs_submitted": 0, "synthetic_ownership_and_identity_derivations": True,
        "actual_gpu_qualification_credit": False, "worker_admission": False}
    path = ROOT / ".review_metadata" / (arguments.label + "_" + record["job_id"] + "_" + record["step_id"] + ".json")
    with path.open("x") as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"receipt": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                      "passed": result.wasSuccessful(), "tests": result.testsRun, "seconds": record["seconds"]}))
    if not result.wasSuccessful():
        print(output.getvalue())
        raise SystemExit(1)


if __name__ == "__main__":
    main()
