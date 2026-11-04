"""Run only source and private metadata tests in the existing owned CPU allocation."""
import ctypes
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import unittest

P = Path(__file__).resolve().parent


def main():
    """Refuse numerical/device/network/process work and retain actual frozen source-test receipts."""
    if os.environ.get("SLURM_JOB_ID") != "6783003" or not os.environ.get("SLURM_STEP_ID", "").isdecimal():
        raise ValueError("source tests require the exact owned CPU allocation")
    freeze = json.loads((P / "source_freeze.json").read_bytes())
    def current():
        return {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in freeze["files"]}
    if current() != freeze["files"]: raise ValueError("raw-review source changed before tests")
    forbidden = {"jax", "jaxlib", "numpy", "torch", "flax", "optax", "cupy", "scipy", "jaxtari", "exploration_platform"}
    def audit(event, values):
        if event == "subprocess.Popen" or event.startswith("socket.") and event != "socket.gethostname":
            raise RuntimeError("no process or network action in source tests")
        if event in {"ctypes.dlopen", "ctypes.dlsym"}:
            raise RuntimeError("no library operation in source tests")
        if event == "import" and values[0].split(".")[0] in forbidden:
            raise RuntimeError("no numerical import in source tests")
        if event == "open" and str(values[0]).startswith(("/dev/nvidia", "/dev/dri", "/tmp/sl5nw_serval02_card1.lock")):
            raise RuntimeError("no device or standing-lock access in source tests")
        if event == "open" and str(values[0]).endswith("/result.json") and "private" not in str(values[0]):
            raise RuntimeError("no actual correctness result open in source tests")
    sys.addaudithook(audit)
    spec = importlib.util.spec_from_file_location("raw_review_metadata_tests", P / "test_review_actual_common_correctness.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(module))
    if current() != freeze["files"]: raise ValueError("raw-review source changed during tests")
    record = {"schema_version": 1, "kind": "owned_cpu_h100_common_correctness_raw_review_source_tests",
        "passed": result.wasSuccessful(), "tests": result.testsRun, "errors": len(result.errors), "failures": len(result.failures),
        "job_id": "6783003", "step_id": os.environ["SLURM_STEP_ID"], "at_utc": datetime.now(timezone.utc).isoformat(),
        "source_files": freeze["files"], "actual_correctness_result_opened": False, "actual_audit_or_review_written": False,
        "scientific_validators_mocked_only_in_private_field_contract_tests": True,
        "host_contact": False, "gpu_operations": 0, "scientific_credit": 0}
    path = P / ("author_source_tests_" + os.environ["SLURM_STEP_ID"] + ".json")
    with path.open("x") as stream:
        json.dump(record, stream, indent=2, sort_keys=True); stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    print(json.dumps({"path": str(path), "passed": result.wasSuccessful(), "tests": result.testsRun}))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
