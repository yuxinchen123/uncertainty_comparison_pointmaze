"""Run the four corrections and unchanged old-reader checks without repeating the broad batch."""
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
SCRIPTS = ROOT / "10_jax_exploration_platform/scripts/atari_pixel_campaign"
FORBIDDEN = {"jax", "jaxlib", "numpy", "torch", "flax", "optax", "scipy", "cupy", "jaxtari", "exploration_platform"}
STARTED = time.monotonic()
STEP = os.environ["SLURM_STEP_ID"]
assert os.environ.get("SLURM_JOB_ID") == "6783003" and STEP.isdecimal() and os.getresuid() == (2618919, 2618919, 2618919)
assert Path("/proc/self/cgroup").read_text().splitlines() == ["0::/system.slice/slurmstepd.scope/job_6783003/step_" + STEP + "/user/task_0"]


def audit(event, values):
    if event == "import" and values[0].split(".")[0] in FORBIDDEN:
        raise RuntimeError("numerical import refused")
    if event in {"socket.connect", "socket.bind", "socket.getaddrinfo", "ctypes.dlopen", "ctypes.dlsym", "os.system"}:
        raise RuntimeError("network or library operation refused")
    if event == "open" and isinstance(values[0], (str, bytes)) and os.fsdecode(values[0]).startswith(("/dev/nvidia", "/dev/dri", "/tmp/sl5nw_serval02_card1.lock")):
        raise RuntimeError("device or standing-lock access refused")
    if event == "subprocess.Popen":
        command = values[1]
        if not (type(command) is list and len(command) == 5 and command[:4] == ["git", "-C", str(ROOT), "show"]
                and command[4].split(":", 1)[0] in {"7fef164909b5eb8b31636da7ae0c653282388203", "0a880984b4a03a2ff7335337cc18576628bc9fc1"}
                and command[4].split(":", 1)[-1] in {"10_jax_exploration_platform/scripts/atari_pixel_campaign/" + name
                    for name in ("direct_measurement.py", "measure_selected.py", "serval02_card1_binding.py")}):
            raise RuntimeError("only exact old source inspection is allowed")


sys.addaudithook(audit)
modules = {}
for name in ("test_accepted_evidence_snapshot", "test_accepted_evidence_snapshot_corrections"):
    specification = importlib.util.spec_from_file_location(name, SCRIPTS / "tests" / (name + ".py"))
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module); modules[name] = module
suite = unittest.defaultTestLoader.loadTestsFromTestCase(modules["test_accepted_evidence_snapshot_corrections"].CorrectedBoundaryTests)
for method in ("test_complete_original_qualification_and_history_bodies_are_retained", "test_numerical_compute_function_is_byte_syntax_unchanged"):
    suite.addTest(modules["test_accepted_evidence_snapshot"].CurrentPlanTests(method))
capture = io.StringIO()
result = unittest.TextTestRunner(stream=capture, verbosity=2).run(suite)
files = [SCRIPTS / name for name in ("accepted_evidence_snapshot.py", "direct_common_confirmation.py", "direct_measurement.py", "measure_selected.py",
    "serval02_card1_binding.py", "tests/test_accepted_evidence_snapshot.py", "tests/test_accepted_evidence_snapshot_corrections.py")]
receipt = {"schema_version": 1, "kind": "targeted_four_snapshot_corrections_author_checks", "job_id": "6783003", "step_id": STEP,
    "checks": result.testsRun, "failures": len(result.failures), "errors": len(result.errors), "passed": result.wasSuccessful(),
    "elapsed_seconds": time.monotonic() - STARTED, "source_files": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
    "previous_broad_34_batch_repeated": False, "actual_large_payload_snapshot_run": False, "gpu_cases_reexecuted": False,
    "preaudit_operations": "Standard-library imports and exact own process/CPU-step checks only.", "numerical_imports": sorted(FORBIDDEN.intersection(sys.modules)),
    "gpu_operations": 0, "network_operations": 0, "scientific_credit": 0}
for suffix, raw in ((".stdout.raw", capture.getvalue().encode()), (".json", (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode())):
    path = Path(__file__).with_name("targeted_author_" + STEP + suffix)
    with path.open("xb") as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
print(json.dumps(receipt, sort_keys=True))
if not result.wasSuccessful():
    print(capture.getvalue(), file=sys.stderr)
raise SystemExit(0 if result.wasSuccessful() else 1)
