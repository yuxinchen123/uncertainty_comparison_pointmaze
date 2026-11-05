"""Run focused API, diagnostic-order and formatter metadata fixtures on the existing owned CPU allocation."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parent.parent
FREEZE = ROOT / ".review_metadata/nsight_interpreter_api_precheck_full_source_freeze.json"


def digest(path):
    """Hash one ordinary metadata source file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    """Require actual owned CPU execution and forbid numerical, tool, device and network operations."""
    if os.environ.get("SLURM_JOB_ID") != "6783003" or not os.environ.get("SLURM_STEP_ID", "").isdigit():
        raise ValueError("an actual numeric step of owned CPU monitor 6783003 is required")
    receipt = Path(sys.argv[1])
    expected = sys.argv[2]
    if os.path.lexists(receipt) or digest(FREEZE) != expected:
        raise ValueError("preserve an earlier receipt and require the exact full source freeze")
    frozen = json.loads(FREEZE.read_text())["files"]
    before = {name: digest(Path(name)) for name in frozen}
    if before != frozen:
        raise ValueError("a source or test changed before the API metadata checks")
    def audit(event, values):
        """Permit private local Git fixtures and reject every actual tool, process, signal or device operation."""
        if event == "subprocess.Popen" and (not isinstance(values[1], (list, tuple)) or values[1][0] != "git"):
            raise RuntimeError("only private local Git fixture commands are permitted")
        if event in {"socket.__new__", "os.system", "os.posix_spawn", "os.kill", "signal.pidfd_send_signal"}:
            raise RuntimeError("network, shell and actual signal operations are forbidden")
        if event == "ctypes.dlopen" and values[0] is not None:
            raise RuntimeError("loading a named native library is forbidden")
        if event == "import" and str(values[0]).split(".")[0] in {"jax", "jaxlib", "numpy", "torch", "flax", "optax", "cupy", "jaxtari"}:
            raise RuntimeError("numerical imports are forbidden")
        if event == "open" and isinstance(values[0], str) and values[0].startswith("/dev/nvidia"):
            raise RuntimeError("GPU device access is forbidden")
    sys.addaudithook(audit)
    scripts = ROOT / "10_jax_exploration_platform/scripts/atari_pixel_campaign"
    sys.path.insert(0, str(scripts))
    modules = {}
    for filename in ("test_nsight_process_control_capability.py", "test_nsight_diagnostic_preparation.py",
                     "test_nsight_process_diagnostic.py"):
        spec = importlib.util.spec_from_file_location("focused_" + filename[:-3], scripts / "tests" / filename)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules[filename] = module
    suite = unittest.TestSuite()
    suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(modules["test_nsight_process_control_capability.py"]))
    suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(modules["test_nsight_diagnostic_preparation.py"]))
    old = modules["test_nsight_process_diagnostic.py"]
    for name in ("test_changed_tool_bytes_refuse_before_even_the_version_subprocess",
                 "test_tool_hash_is_rechecked_after_the_recorded_version_return"):
        suite.addTest(old.DiagnosticTests(name))
    def identities(item):
        """Retain every fixture identity so duplicate or missing checks are refused."""
        return [name for child in item for name in identities(child)] if isinstance(item, unittest.TestSuite) else [item.id()]
    names = identities(suite)
    if len(names) != 24 or len(set(names)) != 24:
        raise ValueError("exactly twenty-four focused metadata cases are required")
    output = io.StringIO()
    started = datetime.now(timezone.utc).isoformat()
    result = unittest.TextTestRunner(stream=output, verbosity=2).run(suite)
    after = {name: digest(Path(name)) for name in frozen}
    value = {"schema_version": 1, "kind": "focused_nsight_interpreter_api_precheck_metadata_tests",
        "passed": result.wasSuccessful() and before == after, "source_before": before, "source_after": after,
        "source_freeze": {"path": str(FREEZE), "sha256": expected}, "test_ids": names, "tests_run": result.testsRun,
        "failures": len(result.failures), "errors": len(result.errors), "test_output": output.getvalue(),
        "job_id": os.environ["SLURM_JOB_ID"], "step_id": os.environ["SLURM_STEP_ID"], "pid": os.getpid(),
        "uid": os.getuid(), "started_at_utc": started, "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "actual_tool_payload_signal_or_gpu_operations": 0, "actual_scheduler_queries_or_job_actions": 0,
        "numerical_imports": False, "scientific_credit": 0}
    with receipt.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    print(output.getvalue(), end="")
    print(json.dumps({"passed": value["passed"], "receipt": str(receipt), "sha256": digest(receipt),
                      "step_id": value["step_id"], "tests": result.testsRun}))
    return 0 if value["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

