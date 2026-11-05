"""Run the focused private reporting cases on exact owned CPU compute only."""
import ast
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import socket
import sys
import unittest

DIRECTORY = Path(__file__).resolve(strict=True).parent
WORKTREE = DIRECTORY.parents[1]
SCRIPT_ROOT = WORKTREE / "10_jax_exploration_platform/scripts/atari_pixel_campaign"
RUN = Path("/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/"
    "2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_"
    "ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1")


def main():
    """Keep the real campaign and every numerical/device/process operation outside this test scope."""
    # Actual allocation, ownership and frozen source bytes precede importing the private test code.
    job, step = os.environ.get("SLURM_JOB_ID"), os.environ.get("SLURM_STEP_ID")
    assert job == "6783003" and re.fullmatch(r"[0-9]+", step or "")
    assert job in (RUN / "slurm/submitted_jobids_uva.txt").read_text().splitlines()
    assert not socket.gethostname().split(".")[0].startswith(("portal", "login"))
    cgroup = Path("/proc/self/cgroup").read_text()
    assert re.search(r"/job_" + job + r"/step_" + step + r"(?:/|$)", cgroup)
    freeze_path = DIRECTORY / "source_freeze_before_author_tests.json"
    freeze = json.loads(freeze_path.read_text())
    assert hashlib.sha256(freeze_path.read_bytes()).hexdigest() == "aaa415325ef4fec8311b1aa3dce972a71a34f6c6fce84885e1cf600db8b1fa77"
    for relative, row in freeze["files"].items():
        assert hashlib.sha256((WORKTREE / relative).read_bytes()).hexdigest() == row["sha256"]
    denied = []

    def audit(event, args):
        """Refuse numerical imports, process/network/library actions and actual campaign reads."""
        forbidden = event in {"subprocess.Popen", "os.system", "os.posix_spawn", "ctypes.dlopen", "ctypes.dlsym", "socket.connect"}
        if event == "import" and args and str(args[0]).split(".")[0] in {"jax", "jaxlib", "numpy", "torch"}:
            forbidden = True
        if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
            path = os.fsdecode(args[0])
            forbidden |= path.startswith((str(RUN) + "/", "/dev/nvidia", "/dev/dri", "/p/rlprojects/RND/"))
        if forbidden:
            denied.append({"event": event, "arguments": repr(args)[:500]})
            raise RuntimeError("operation outside metadata reporting tests: " + event)

    sys.addaudithook(audit)
    sys.path.insert(0, str(SCRIPT_ROOT / "tests"))
    sys.path.insert(0, str(SCRIPT_ROOT))
    specification = importlib.util.spec_from_file_location("private_post_initial_reporting_tests",
        SCRIPT_ROOT / "tests/test_post_initial_phase_reporting.py")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    existing = module.queue_fixtures.QueueProtocolTests
    for name in ("test_thirty_completed_ideas_leave_twenty_unchosen",
                 "test_additional_cohort_can_have_another_source_and_copy_grid",
                 "test_initial_completion_waits_for_every_matched_copy_count"):
        suite.addTest(existing(name))
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    assert all(hashlib.sha256((WORKTREE / relative).read_bytes()).hexdigest() == row["sha256"]
               for relative, row in freeze["files"].items())
    record = {"kind": "post_initial_reporting_author_metadata_checks", "passed": result.wasSuccessful() and not denied,
              "job_id": job, "step_id": step, "host": socket.gethostname(), "uid": os.getuid(),
              "source_freeze": {"path": str(freeze_path), "sha256": hashlib.sha256(freeze_path.read_bytes()).hexdigest()},
              "method_executions": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
              "denied_events": denied, "stdout": stream.getvalue(), "scientific_credit": 0,
              "actual_numerical_or_gpu_or_library_or_network_or_process_actions": 0,
              "live_campaign_or_monitor_changed": False}
    path = DIRECTORY / ("author_" + job + "_" + step + ".json")
    with path.open("x") as output:
        output.write(json.dumps(record, indent=2) + "\n")
    print(stream.getvalue(), file=sys.stderr)
    print(json.dumps({"passed": record["passed"], "method_executions": result.testsRun,
                      "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}))
    return 0 if record["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
