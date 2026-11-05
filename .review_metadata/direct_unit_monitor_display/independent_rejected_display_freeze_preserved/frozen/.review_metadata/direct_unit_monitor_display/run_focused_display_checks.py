"""Check direct metadata display and existing six reviewed reporting entries on owned CPU."""
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
CAMPAIGN = Path("/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1")
STARTED = time.monotonic(); STEP = os.environ["SLURM_STEP_ID"]
assert os.environ.get("SLURM_JOB_ID") == "6783003" and STEP.isdecimal() and os.getresuid() == (2618919, 2618919, 2618919)
assert Path("/proc/self/cgroup").read_text().splitlines() == ["0::/system.slice/slurmstepd.scope/job_6783003/step_" + STEP + "/user/task_0"]
FORBIDDEN = {"jax", "jaxlib", "numpy", "torch", "flax", "optax", "scipy", "cupy", "jaxtari", "exploration_platform"}


def audit(event, values):
    if event == "import" and values[0].split(".")[0] in FORBIDDEN:
        raise RuntimeError("no numerical import in display checks")
    if event in {"socket.connect", "socket.bind", "socket.getaddrinfo", "ctypes.dlopen", "ctypes.dlsym", "os.system", "subprocess.Popen"}:
        raise RuntimeError("no network, library or tool operation in display checks")
    if event == "open" and isinstance(values[0], (str, bytes)) and os.fsdecode(values[0]).startswith(("/dev/nvidia", "/dev/dri", "/tmp/sl5nw_serval02_card1.lock")):
        raise RuntimeError("no GPU or standing-lock operation in display checks")


sys.addaudithook(audit)
test_path = SCRIPTS / "tests/test_monitor_direct_unit_display.py"
specification = importlib.util.spec_from_file_location("focused_direct_display_tests", test_path)
module = importlib.util.module_from_spec(specification); specification.loader.exec_module(module)
capture = io.StringIO(); result = unittest.TextTestRunner(stream=capture, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(module))
sys.path.insert(0, str(SCRIPTS))
import controller
actions = json.loads((CAMPAIGN / "next_actions.json").read_bytes()); index = actions["post_initial_reporting_evidence"]
assert sum(len(index.get(name, [])) for name in ("common_correctness", "common_confirmation", "kernel_profiles", "packing")) == 6
reporting = controller.post_initial_reporting_requirements(CAMPAIGN, index["source_commit"], [1, 2], 1)
assert reporting[0]["state"] == "reviewed_evidence_present"
assert reporting[1]["missing"] == [{"hardware_class": "h100", "copies": 1}, {"hardware_class": "h100", "copies": 2}]
assert all(row["scientific_credit"] == 0 and row["admission_authority"] is False for row in reporting)
receipt = {"schema_version": 1, "kind": "focused_direct_display_and_existing_reporting_metadata_checks", "job_id": "6783003", "step_id": STEP,
    "base_commit": "6b7d492217140bba3d2f36b93754ecde2d020ccc", "focused_tests": result.testsRun, "passed": result.wasSuccessful(),
    "failures": len(result.failures), "errors": len(result.errors), "elapsed_seconds": time.monotonic() - STARTED,
    "source_files": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in (SCRIPTS / "monitor.py", test_path)},
    "current_six_accepted_entries_retained": True, "post_initial_reporting_requirements": reporting,
    "new_monitor_job_submitted": False, "old_monitor_stopped": False, "full_runtime_verification_claimed": False,
    "broad1540_checks_repeated": False, "preaudit_operations": "Standard-library imports and exact own CPU-step/process checks only.",
    "numerical_imports": sorted(FORBIDDEN.intersection(sys.modules)), "gpu_operations": 0, "network_operations": 0, "scientific_credit": 0}
for suffix, raw in ((".json", (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode()), (".stdout.raw", capture.getvalue().encode())):
    with Path(__file__).with_name("focused_display_" + STEP + suffix).open("xb") as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
print(json.dumps({key: value for key, value in receipt.items() if key != "post_initial_reporting_requirements"}, sort_keys=True))
if not result.wasSuccessful():
    print(capture.getvalue(), file=sys.stderr)
raise SystemExit(0 if result.wasSuccessful() else 1)
