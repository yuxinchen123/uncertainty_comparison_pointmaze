"""Check explicit placement using private Git fixtures on the owned CPU allocation only."""
import datetime
import hashlib
import importlib.abc
import importlib.util
import json
import os
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve(strict=True).parent
WORKTREE = HERE.parent.parent
FORBIDDEN = {"jax", "jaxlib", "numpy", "torch", "flax", "optax", "cupy", "scipy", "jaxtari"}


def require(value, message):
    """Stop rather than running outside the declared metadata scope."""
    if not value:
        raise ValueError(message)


def source_records():
    """Reconstruct the exact two changed source records before and after the tests."""
    freeze = json.loads((HERE / "source_freeze.json").read_text())
    result = {}
    for name in freeze["files"]:
        path = WORKTREE / name
        result[name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "bytes": path.stat().st_size, "mode": oct(path.stat().st_mode & 0o777)}
    require(result == freeze["files"], "the frozen formatter or its tests changed")
    return result


class NoNumericalImports(importlib.abc.MetaPathFinder):
    """Keep every test independent of scientific runtimes."""
    def find_spec(self, fullname, path=None, target=None):
        """Refuse an array or learner package before import."""
        if fullname.split(".")[0] in FORBIDDEN:
            raise RuntimeError("numerical packages are forbidden in formatter metadata checks")


def install_scope_guard(commands):
    """Permit local Git fixtures while refusing tool, network, library and device actions."""
    require(not FORBIDDEN.intersection(sys.modules), "a numerical package was already imported")
    sys.meta_path.insert(0, NoNumericalImports())

    def audit(event, values):
        """Retain local Git commands and refuse any broader native operation."""
        if event in {"os.system", "socket.connect", "socket.bind", "socket.getaddrinfo", "ctypes.dlopen", "ctypes.dlsym"}:
            raise RuntimeError("metadata check attempted a forbidden native operation: " + event)
        if event == "open" and isinstance(values[0], str):
            require(not values[0].startswith(("/dev/nvidia", "/dev/dri")), "metadata checks cannot open a GPU device")
        if event == "subprocess.Popen":
            argv = values[1]
            require(isinstance(argv, list) and argv and argv[0] == "git" and "-C" in argv,
                    "only exact local Git fixture commands are permitted")
            require(not any(value in argv for value in ("push", "pull", "fetch", "clone", "ls-remote")),
                    "metadata fixtures cannot contact a remote repository")
            commands.append(argv)

    sys.addaudithook(audit)


def flattened(suite):
    """Retain each actual test method without discovering imported duplicate cases."""
    for entry in suite:
        if isinstance(entry, unittest.TestSuite):
            yield from flattened(entry)
        else:
            yield entry


def main():
    """Run the focused metadata suite once and write its actual result and step identity."""
    job, step = os.environ.get("SLURM_JOB_ID"), os.environ.get("SLURM_STEP_ID")
    require(job == "6783003" and type(step) is str and step.isdigit(),
            "formatter checks require the current owned CPU monitor numeric step")
    before = source_records()
    commands = []
    install_scope_guard(commands)
    test_path = WORKTREE / "10_jax_exploration_platform/scripts/atari_pixel_campaign/tests/test_nsight_diagnostic_preparation.py"
    specification = importlib.util.spec_from_file_location("explicit_placement_metadata_tests", test_path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    tests = list(flattened(unittest.defaultTestLoader.loadTestsFromModule(module)))
    identifiers = [test.id() for test in tests]
    require(len(identifiers) == len(set(identifiers)) == 12, "unexpected focused test count or duplicate IDs")
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(tests))
    after = source_records()
    receipt = {"kind": "explicit_nsight_reservation_formatter_author_metadata_tests",
               "schema_version": 1, "passed": result.wasSuccessful(), "job_id": job, "step_id": step,
               "started_at_utc": started, "finished_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
               "hostname": os.uname().nodename, "pid": os.getpid(), "tests_run": result.testsRun,
               "unique_test_ids": identifiers, "failures": [(test.id(), text) for test, text in result.failures],
               "errors": [(test.id(), text) for test, text in result.errors],
               "source_before": before, "source_after": after, "local_git_commands": commands,
               "actual_tool_or_gpu_or_network_operations": 0, "actual_campaign_packets_prepared": 0,
               "actual_jobs_submitted": 0, "scientific_credit": 0}
    path = HERE / ("author_" + job + "_" + step + ".json")
    with path.open("x") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    print(json.dumps({"passed": result.wasSuccessful(), "receipt": str(path),
                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "tests_run": result.testsRun}))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
