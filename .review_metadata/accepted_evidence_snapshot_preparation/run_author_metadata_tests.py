"""Run only small fixtures in the exact owned CPU step and preserve every actual result."""
import ast
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "10_jax_exploration_platform/scripts/atari_pixel_campaign"
OUTPUT = Path(__file__).parent
FORBIDDEN = {"jax", "jaxlib", "numpy", "torch", "flax", "optax", "scipy", "cupy", "jaxtari", "exploration_platform"}
STARTED = time.monotonic()


def audit(event, values):
    if event == "import" and values[0].split(".")[0] in FORBIDDEN:
        raise RuntimeError("numerical import refused in metadata fixture check")
    if event in {"socket.connect", "socket.bind", "socket.getaddrinfo", "ctypes.dlopen", "ctypes.dlsym", "os.system"}:
        raise RuntimeError("network or library operation refused in metadata fixture check")
    if event == "open" and isinstance(values[0], (str, bytes)) and os.fsdecode(values[0]).startswith(("/dev/nvidia", "/dev/dri", "/tmp/sl5nw_serval02_card1.lock")):
        raise RuntimeError("GPU device or standing lock access refused in metadata fixture check")
    if event == "subprocess.Popen":
        command = values[1]
        if not (type(command) is list and len(command) == 5 and command[:4] == ["git", "-C", str(ROOT), "show"]
                and command[4].split(":", 1)[0] in {"7fef164909b5eb8b31636da7ae0c653282388203", "0a880984b4a03a2ff7335337cc18576628bc9fc1"}
                and command[4].split(":", 1)[-1] in {"10_jax_exploration_platform/scripts/atari_pixel_campaign/" + name
                    for name in ("direct_measurement.py", "measure_selected.py", "serval02_card1_binding.py")}):
            raise RuntimeError("only exact old source blob inspection is allowed")


job, step = os.environ.get("SLURM_JOB_ID"), os.environ.get("SLURM_STEP_ID")
assert job == "6783003" and type(step) is str and step.isdecimal()
assert Path("/proc/self/cgroup").read_text().splitlines() == ["0::/system.slice/slurmstepd.scope/job_6783003/step_" + step + "/user/task_0"]
assert os.getuid() == os.geteuid() == 2618919
sys.addaudithook(audit)
specification = importlib.util.spec_from_file_location("accepted_snapshot_author_tests", SCRIPTS / "tests/test_accepted_evidence_snapshot.py")
module = importlib.util.module_from_spec(specification)
specification.loader.exec_module(module)
suite = unittest.defaultTestLoader.loadTestsFromModule(module)
capture = io.StringIO()
result = unittest.TextTestRunner(stream=capture, verbosity=2).run(suite)
source_files = [SCRIPTS / name for name in ("accepted_evidence_snapshot.py", "direct_common_confirmation.py", "direct_measurement.py", "measure_selected.py", "serval02_card1_binding.py", "tests/test_accepted_evidence_snapshot.py")]
for path in source_files:
    compile(path.read_bytes(), str(path), "exec")
receipt = {"schema_version": 1, "kind": "accepted_evidence_snapshot_author_metadata_check", "job_id": job, "step_id": step,
    "pid": os.getpid(), "checks_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
    "passed": result.wasSuccessful(), "elapsed_seconds": time.monotonic() - STARTED,
    "source_files": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files},
    "preaudit_operations": "Standard-library imports and exact own process/CPU-step identity checks only.",
    "numerical_imports": sorted(FORBIDDEN.intersection(sys.modules)), "gpu_operations": 0, "network_operations": 0,
    "actual_historical_payload_snapshot_run": False, "scientific_credit": 0}
stem = "author_metadata_check_" + job + "_" + step
for name, raw in ((stem + ".stdout.raw", capture.getvalue().encode()), (stem + ".json", (json.dumps(receipt, sort_keys=True, indent=2) + "\n").encode())):
    with (OUTPUT / name).open("xb") as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
print(json.dumps(receipt, sort_keys=True))
if not result.wasSuccessful():
    print(capture.getvalue(), file=sys.stderr)
raise SystemExit(0 if result.wasSuccessful() else 1)
