"""Independently check the narrow reservation-dependent quality-of-service correction."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import unittest

DIRECTORY = Path(__file__).resolve().parent
EXPECTED_FREEZE = "8b7abd5698e94da9ec85d165562db913cc3fd64abca57f57a3876c633df88dfe"


def main():
    """Run the twelve private-file metadata tests without actual scheduler, tool or numerical work."""
    if os.environ.get("SLURM_JOB_ID") != "6783003" or not os.environ.get("SLURM_STEP_ID", "").isdigit():
        raise ValueError("an actual numeric step of owned CPU monitor 6783003 is required")
    freeze_path = DIRECTORY / "source_freeze.json"
    if hashlib.sha256(freeze_path.read_bytes()).hexdigest() != EXPECTED_FREEZE:
        raise ValueError("the reviewed two-file source freeze changed")
    frozen = json.loads(freeze_path.read_text())["files"]
    before = {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in frozen}
    if before != frozen:
        raise ValueError("the actual formatter or test source changed")
    def audit(event, values):
        """Permit only private local Git fixture subprocesses and refuse numerical or tool actions."""
        if event == "subprocess.Popen" and (not isinstance(values[1], (list, tuple)) or values[1][0] != "git"):
            raise RuntimeError("only private local Git commands are permitted")
        if event in {"socket.__new__", "ctypes.dlopen", "os.system", "os.posix_spawn"}:
            raise RuntimeError("network, native library and shell work are forbidden")
        if event == "import" and str(values[0]).split(".")[0] in {"jax", "jaxlib", "numpy", "torch", "cupy", "pynvml"}:
            raise RuntimeError("numerical imports are forbidden")
    sys.addaudithook(audit)
    test_path = next(Path(name) for name in frozen if Path(name).name.startswith("test_"))
    spec = importlib.util.spec_from_file_location("independent_nsight_placement_tests", test_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    def names(item):
        """Flatten all test identities and retain any duplicate for refusal."""
        return [name for child in item for name in names(child)] if isinstance(item, unittest.TestSuite) else [item.id()]
    ids = names(suite)
    if len(ids) != 12 or len(set(ids)) != 12:
        raise ValueError("exactly twelve distinct metadata tests are required")
    started = datetime.now(timezone.utc).isoformat()
    output = io.StringIO()
    result = unittest.TextTestRunner(stream=output, verbosity=2).run(suite)
    after = {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in frozen}
    value = {"schema_version": 1, "passed": result.wasSuccessful() and before == after,
        "kind": "independent_nsight_reservation_dependent_qos_metadata_review",
        "source_freeze": {"path": str(freeze_path), "sha256": EXPECTED_FREEZE},
        "source_before": before, "source_after": after, "test_ids": ids, "tests_run": result.testsRun,
        "failures": len(result.failures), "errors": len(result.errors), "test_output": output.getvalue(),
        "job_id": os.environ["SLURM_JOB_ID"], "step_id": os.environ["SLURM_STEP_ID"], "uid": os.getuid(),
        "started_at_utc": started, "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "independent_source_review": "The only formatter change selects normal for an explicitly unreserved request and csresnolim for an explicit reservation. The resolved plan and held command use the same selected value. Six added assertions check both plan values and both command inclusions and exclusions. The actual recorded association permits both; csresnolim requires a reservation. Producer source and numerical work are unchanged.",
        "actual_scheduler_actions": 0, "actual_tool_executions": 0, "numerical_imports": False,
        "scientific_credit": 0}
    path = DIRECTORY / "independent_review.json"
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True); stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    print(output.getvalue(), end="")
    print(json.dumps({"passed": value["passed"], "step_id": value["step_id"], "receipt": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}))
    return 0 if value["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

