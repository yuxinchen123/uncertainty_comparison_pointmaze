"""Check only the newly added exact controller metadata binding display."""
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parents[2]; STEP = os.environ["SLURM_STEP_ID"]; STARTED = time.monotonic()
assert os.environ.get("SLURM_JOB_ID") == "6783003" and STEP.isdecimal() and os.getresuid() == (2618919, 2618919, 2618919)
assert Path("/proc/self/cgroup").read_text().splitlines() == ["0::/system.slice/slurmstepd.scope/job_6783003/step_" + STEP + "/user/task_0"]
FORBIDDEN = {"jax", "jaxlib", "numpy", "torch", "flax", "optax", "scipy", "cupy", "jaxtari", "exploration_platform"}


def audit(event, values):
    if event == "import" and values[0].split(".")[0] in FORBIDDEN:
        raise RuntimeError("no numerical imports")
    if event in {"socket.connect", "socket.bind", "socket.getaddrinfo", "ctypes.dlopen", "ctypes.dlsym", "os.system", "subprocess.Popen"}:
        raise RuntimeError("no network, library or tool actions")
    if event == "open" and isinstance(values[0], (str, bytes)) and os.fsdecode(values[0]).startswith(("/dev/nvidia", "/dev/dri", "/tmp/sl5nw_serval02_card1.lock")):
        raise RuntimeError("no device or standing lock access")


sys.addaudithook(audit)
path = ROOT / "10_jax_exploration_platform/scripts/atari_pixel_campaign/tests/test_monitor_direct_unit_display.py"
specification = importlib.util.spec_from_file_location("affected_monitor_fixture", path)
module = importlib.util.module_from_spec(specification); specification.loader.exec_module(module)
capture = io.StringIO(); result = unittest.TextTestRunner(stream=capture, verbosity=2).run(unittest.TestSuite([
    module.DirectDisplayTests("test_controller_metadata_binding_does_not_claim_registration_or_running")]))
receipt = {"schema_version": 1, "kind": "direct_controller_binding_display_check", "job_id": "6783003", "step_id": STEP,
    "methods": result.testsRun, "passed": result.wasSuccessful(), "errors": len(result.errors), "failures": len(result.failures),
    "elapsed_seconds": time.monotonic() - STARTED, "monitor_source_sha256": hashlib.sha256((path.parents[1] / "monitor.py").read_bytes()).hexdigest(),
    "test_source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "source_logic_changed_after_first_check": True,
    "old_other_six_methods_or_reporting_join_repeated": False, "numerical_imports": sorted(FORBIDDEN.intersection(sys.modules)),
    "gpu_operations": 0, "network_operations": 0, "scientific_credit": 0}
for suffix, raw in ((".json", (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode()), (".stdout.raw", capture.getvalue().encode())):
    with Path(__file__).with_name("controller_binding_display_" + STEP + suffix).open("xb") as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
print(json.dumps(receipt, sort_keys=True))
if not result.wasSuccessful():
    print(capture.getvalue(), file=sys.stderr)
raise SystemExit(0 if result.wasSuccessful() else 1)
