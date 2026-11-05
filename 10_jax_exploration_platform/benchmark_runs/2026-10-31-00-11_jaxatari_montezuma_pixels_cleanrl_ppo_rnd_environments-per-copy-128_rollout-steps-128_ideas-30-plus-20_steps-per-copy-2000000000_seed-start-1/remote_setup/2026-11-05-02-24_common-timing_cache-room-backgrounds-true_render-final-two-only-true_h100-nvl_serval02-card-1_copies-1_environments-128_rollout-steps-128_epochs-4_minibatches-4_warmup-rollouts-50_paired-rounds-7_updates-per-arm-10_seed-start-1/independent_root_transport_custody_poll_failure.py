"""Reproduce one ordinary custody-poll error on a real CPU-only child."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import time
from unittest.mock import patch

P = Path(__file__).resolve().parent
R = P.parent.parent


def load(name, path):
    """Load exact frozen metadata source without invoking its launch entry."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(path):
    """Bind one small frozen source or metadata file to its actual bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# Refuse any changing source before executing the one affected CPU case.
assert os.environ["SLURM_JOB_ID"] == "6821437" and os.environ["SLURM_STEP_ID"].isdigit()
assert os.getresuid() == (2618919, 2618919, 2618919)
assert (R / "slurm/submitted_jobids_uva.txt").read_text().splitlines().count("6821437") == 1
freeze_path = P / "corrected_root_transport_source_freeze.json"
assert digest(freeze_path) == "b76c23df24fcac60fc1afd2524c8249edf588b95624ccb1757deca5dc72f0a65"
freeze = json.loads(freeze_path.read_text())
for path, expected in freeze["files"].items():
    assert digest(path) == expected
starter = load("reviewed_root_starter_failure_case", P / "root_start_controller_once.py")
control = load("unchanged_metadata_controller_for_failure_case", P / "direct_qualification_controller.py")
started = time.monotonic()

# Keep an actual owned child waiting on a private metadata lock; no GPU or numerical code exists here.
with tempfile.TemporaryDirectory(prefix="root-transport-custody-cpu-") as temporary:
    root = Path(temporary)
    lock = root / "metadata.lock"
    lock.write_text("")
    lock.chmod(0o600)
    child_script = root / "only_metadata_lock.py"
    child_script.write_text("import fcntl,pathlib,sys\nr=pathlib.Path(sys.argv[1])\n(r/'before_lock').write_text('actual CPU child reached lock')\nwith (r/'metadata.lock').open('r+') as f:\n fcntl.flock(f,fcntl.LOCK_EX)\n (r/'after_lock').write_text('actual CPU child passed barrier without root spawn record')\n")
    argv = ["/usr/bin/python3", "-I", "-S", "-B", str(child_script), str(root)]
    escaped = None
    with control.metadata_lock(lock, 2618919):
        child = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        birth = Path("/proc", str(child.pid), "stat").read_text().rsplit(")", 1)[1].split()[19]
        assert Path("/proc", str(child.pid)).stat().st_uid == 2618919
        deadline = time.monotonic() + 5
        while not (root / "before_lock").exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert (root / "before_lock").exists() and child.poll() is None and not (root / "after_lock").exists()

        class InjectedPollFailure:
            """Keep the actual Popen PID while injecting exactly one ordinary poll exception."""
            pid = child.pid

            def poll(self):
                """Reproduce an ordinary error before any positive child-exit observation."""
                raise OSError("declared CPU fixture: ordinary Popen.poll failure")

            def wait(self):
                """Retain the exact actual Popen wait if the implementation reaches it."""
                return child.wait()

        # The frozen helper escapes on the injected ordinary error while the actual child is alive.
        with patch.object(starter, "P", root):
            try:
                starter.retain_custody(InjectedPollFailure(), RuntimeError("declared unknown identity fixture"))
            except OSError as error:
                escaped = str(error)
        alive_before_unlock = child.poll() is None
        assert escaped is not None and alive_before_unlock and not (root / "after_lock").exists()
    actual_return = child.wait(timeout=5)
    assert actual_return == 0 and (root / "after_lock").exists() and not (root / "root_actual_controller_spawn.json").exists()
    result = {"schema_version": 1, "kind": "independent_root_transport_unknown_custody_poll_exception_reproduction", "reproduced": True,
        "production_source_approved": False, "actual_owned_cpu_job": os.environ["SLURM_JOB_ID"], "actual_numeric_step": os.environ["SLURM_STEP_ID"],
        "hostname": socket.gethostname(), "real_cpu_child_pid": child.pid, "real_cpu_child_birth": birth,
        "injected_failure": "one explicitly declared ordinary Popen.poll OSError", "actual_escaped_error": escaped,
        "real_child_alive_when_custody_helper_escaped": alive_before_unlock, "real_child_passed_metadata_barrier_after_unwind": True,
        "root_spawn_record_existed": False, "actual_cpu_child_wait_returncode": actual_return,
        "source_freeze": {"path": str(freeze_path), "sha256": digest(freeze_path)}, "frozen_starter_sha256": digest(P / "root_start_controller_once.py"),
        "numerical_gpu_tool_remote_operations": 0, "actual_target_operation": False, "scientific_credit": 0, "elapsed_seconds": time.monotonic() - started}
for path, expected in freeze["files"].items():
    assert digest(path) == expected
out = P / "independent_root_transport_custody_poll_failure.json"
with out.open("x") as handle:
    json.dump(result, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
print(json.dumps({"reproduced": True, "production_source_approved": False, "receipt": str(out), "sha256": digest(out), "elapsed_seconds": result["elapsed_seconds"]}))
