"""Run only the five new reference-scope controls in the exact owned CPU step."""
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
STEP = os.environ["SLURM_STEP_ID"]
STARTED = time.monotonic()
FORBIDDEN = {"jax", "jaxlib", "numpy", "torch", "flax", "optax", "scipy", "cupy", "jaxtari", "exploration_platform"}
assert os.environ.get("SLURM_JOB_ID") == "6783003" and STEP.isdecimal() and os.getresuid() == (2618919, 2618919, 2618919)
assert Path("/proc/self/cgroup").read_text().splitlines() == ["0::/system.slice/slurmstepd.scope/job_6783003/step_" + STEP + "/user/task_0"]


def audit(event, values):
    if event == "import" and values[0].split(".")[0] in FORBIDDEN:
        raise RuntimeError("no numerical import")
    if event in {"socket.connect", "socket.bind", "socket.getaddrinfo", "ctypes.dlopen", "ctypes.dlsym", "os.system", "subprocess.Popen"}:
        raise RuntimeError("no network, library or subprocess operation")
    if event == "open" and isinstance(values[0], (str, bytes)) and os.fsdecode(values[0]).startswith(("/dev/nvidia", "/dev/dri", "/tmp/sl5nw_serval02_card1.lock")):
        raise RuntimeError("no GPU device or standing-lock access")


sys.addaudithook(audit)
test_path = ROOT / "10_jax_exploration_platform/scripts/atari_pixel_campaign/tests/test_accepted_evidence_snapshot_reference_scope.py"
specification = importlib.util.spec_from_file_location("new_scope_tests", test_path)
module = importlib.util.module_from_spec(specification)
specification.loader.exec_module(module)
suite = unittest.defaultTestLoader.loadTestsFromModule(module)
capture = io.StringIO(); result = unittest.TextTestRunner(stream=capture, verbosity=2).run(suite)
source = test_path.parents[1] / "accepted_evidence_snapshot.py"
receipt = {"schema_version": 1, "kind": "targeted_frozen_ledger_and_controller_reference_scope_regressions",
    "job_id": "6783003", "step_id": STEP, "checks": result.testsRun, "passed": result.wasSuccessful(),
    "errors": len(result.errors), "failures": len(result.failures), "elapsed_seconds": time.monotonic() - STARTED,
    "candidate_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "test_source_sha256": hashlib.sha256(test_path.read_bytes()).hexdigest(),
    "broad_previous_suites_repeated": False, "preaudit_operations": "Standard-library imports and exact own CPU-step/process checks only.",
    "numerical_imports": sorted(FORBIDDEN.intersection(sys.modules)), "gpu_operations": 0, "network_operations": 0,
    "payload_files_opened": 0, "scientific_credit": 0}
for suffix, raw in ((".json", (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode()), (".stdout.raw", capture.getvalue().encode())):
    with Path(__file__).with_name("scope_regressions_" + STEP + suffix).open("xb") as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
print(json.dumps(receipt, sort_keys=True))
if not result.wasSuccessful():
    print(capture.getvalue(), file=sys.stderr)
raise SystemExit(0 if result.wasSuccessful() else 1)
