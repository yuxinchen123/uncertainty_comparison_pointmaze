"""Run only private transport metadata tests inside the existing owned CPU allocation."""
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
    """Retain actual test identity while refusing all processes, network, numerical and device operations."""
    if os.environ.get("SLURM_JOB_ID") != "6783003" or not os.environ.get("SLURM_STEP_ID", "").isdecimal():
        raise ValueError("only the exact owned CPU allocation may run metadata tests")
    freeze = json.loads((P / "launch_transport_metadata_tests_source_freeze.json").read_bytes())
    def current():
        return {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in freeze["files"]}
    if current() != freeze["files"]: raise ValueError("the frozen metadata writer/test source changed")
    blocked = {"jax", "jaxlib", "numpy", "torch", "flax", "optax", "cupy", "scipy", "jaxtari", "exploration_platform"}
    def audit(event, values):
        if event == "subprocess.Popen" or event.startswith("socket.") and event != "socket.gethostname":
            raise RuntimeError("process/network operation forbidden in metadata writer tests")
        if event in {"ctypes.dlopen", "ctypes.dlsym"}:
            raise RuntimeError("library operation forbidden in metadata writer tests")
        if event == "import" and values[0].split(".")[0] in blocked:
            raise RuntimeError("numerical import forbidden in metadata writer tests")
        if event == "open" and str(values[0]).startswith(("/dev/nvidia", "/dev/dri", "/tmp/sl5nw_serval02_card1.lock")):
            raise RuntimeError("device/standing-lock operation forbidden in metadata writer tests")
    sys.addaudithook(audit)
    spec = importlib.util.spec_from_file_location("private_transport_metadata_tests", P / "test_prepare_root_launch_transport.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(module))
    if current() != freeze["files"]: raise ValueError("metadata writer/test bytes changed during the run")
    record = {"schema_version": 1, "kind": "owned_cpu_private_transport_metadata_author_tests",
        "passed": result.wasSuccessful(), "tests": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
        "job_id": "6783003", "step_id": os.environ["SLURM_STEP_ID"], "at_utc": datetime.now(timezone.utc).isoformat(),
        "source_files": freeze["files"], "host_contact": False, "gpu_operations": 0, "scientific_credit": 0,
        "synthetic_publication_fixture_is_not_actual_approval": True}
    path = P / ("transport_metadata_author_" + os.environ["SLURM_STEP_ID"] + ".json")
    with path.open("x") as stream:
        json.dump(record, stream, indent=2, sort_keys=True); stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    print(json.dumps({"path": str(path), "passed": result.wasSuccessful(), "tests": result.testsRun}))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
