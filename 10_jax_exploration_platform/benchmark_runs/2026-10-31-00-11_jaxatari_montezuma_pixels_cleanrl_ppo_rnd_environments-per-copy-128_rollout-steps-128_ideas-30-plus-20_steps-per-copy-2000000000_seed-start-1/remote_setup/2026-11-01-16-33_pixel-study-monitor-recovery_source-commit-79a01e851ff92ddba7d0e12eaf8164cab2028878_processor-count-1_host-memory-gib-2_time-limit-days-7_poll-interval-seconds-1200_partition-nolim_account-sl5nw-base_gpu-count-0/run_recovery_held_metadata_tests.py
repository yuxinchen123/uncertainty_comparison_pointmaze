"""Only standard-library submission-control metadata checks on portal05."""
import ctypes
from datetime import datetime, timezone
import hashlib
import importlib.abc
import importlib.util
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import unittest

P = Path(__file__).absolute().parent
assert socket.gethostname().split(".")[0] == "portal05" and os.getuid() == 2618919
assert not os.environ.get("SLURM_JOB_ID")
NUM = {"jax", "jaxlib", "numpy", "torch", "flax", "optax", "cupy", "scipy", "jaxtari"}
assert not NUM.intersection(sys.modules)
names = ["submit_monitor_held.py", "test_submit_monitor_held.py", "run_recovery_held_metadata_tests.py",
         "held_submission_inputs.json", "held_submission_author_review.json"]
def images():
    return {name: hashlib.sha256((P / name).read_bytes()).hexdigest() for name in names}
before = images()
inputs = json.loads((P / "held_submission_inputs.json").read_bytes())
protected_before = {entry["path"]: hashlib.sha256(Path(entry["path"]).read_bytes()).hexdigest()
                    for entry in inputs["required_files"]}
assert protected_before == {entry["path"]: entry["sha256"] for entry in inputs["required_files"]}
blocked = []
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in NUM:
            blocked.append(fullname)
            raise AssertionError("numerical imports forbidden")
sys.meta_path.insert(0, Block())
def audit(event, args):
    if event.startswith("socket.") and event != "socket.gethostname":
        raise AssertionError("network access forbidden")
    if event in {"ctypes.dlopen", "ctypes.dlsym"}:
        raise AssertionError("dynamic-library operation forbidden")
    if event == "open" and (str(args[0]).startswith(("/dev/nvidia", "/dev/dri"))
                            or str(args[0]) == "/tmp/sl5nw_serval02_card1.lock"):
        raise AssertionError("GPU or standing-lock operation forbidden")
    if event == "subprocess.Popen":
        raise AssertionError("all live commands forbidden during control tests")
sys.addaudithook(audit)
spec = importlib.util.spec_from_file_location("recovery_held_tests", P / "test_submit_monitor_held.py")
tests = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tests)
output = io.StringIO()
result = unittest.TextTestRunner(stream=output, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(tests))
actual_inputs, plan = tests.submit.validate_inputs()
after = images()
protected_after = {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in protected_before}
assert after == before and protected_after == protected_before and not blocked
value = {"schema_version": 1, "kind": "recovery_held_submission_portal_metadata_verification",
    "at_utc": datetime.now(timezone.utc).isoformat(), "metadata_host": socket.gethostname(),
    "metadata_allocation": None, "pid": os.getpid(), "uid": os.getuid(),
    "proc_stat": Path("/proc/self/stat").read_text(), "tests": result.testsRun,
    "passed": result.wasSuccessful(), "output": output.getvalue(), "source_before": before, "source_after": after,
    "actual_required_inputs": len(actual_inputs["required_files"]),
    "protected_before": protected_before, "protected_after": protected_after,
    "actual_static_input_check": True, "actual_publication_check": False,
    "real_root_publication_approval_still_required": True,
    "prelaunch_full_candidate_available": False, "actual_full_campaign_report_on_login": False,
    "numerical_import_attempts": blocked, "live_commands": 0, "new_jobs": 0, "release_calls": 0,
    "scientific_credit": 0, "worker_admitted": False}
raw = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
target = P / "held_submission_author_portal_metadata.json"
with target.open("xb") as stream:
    stream.write(raw)
    stream.flush()
    os.fsync(stream.fileno())
descriptor = os.open(P, os.O_DIRECTORY)
os.fsync(descriptor)
os.close(descriptor)
print(json.dumps({"receipt": str(target), "sha256": hashlib.sha256(raw).hexdigest(),
                  "tests": result.testsRun, "passed": result.wasSuccessful(), "output": output.getvalue()}))
raise SystemExit(0 if result.wasSuccessful() else 1)

