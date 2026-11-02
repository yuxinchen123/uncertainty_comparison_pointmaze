"""Run only private metadata process tests inside the current owned CPU allocation."""
import ctypes
from datetime import datetime, timezone
import hashlib
import importlib.abc
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys
import time
import unittest

PACKET = Path(__file__).absolute().parent
SOURCE_NAMES = ("direct_qualification_controller.py", "test_direct_qualification_controller.py",
                "run_direct_controller_metadata_tests.py")
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    if os.environ.get("SLURM_JOB_ID") != "6783003" or not os.environ.get("SLURM_STEP_ID", "").isdigit():
        raise ValueError("metadata checks require the actual current owned allocation 6783003 and a numeric step")
    source_before = {name: sha(PACKET / name) for name in SOURCE_NAMES}
    original_freeze_raw = (PACKET / "packet_metadata_freeze.json").read_bytes()
    original_files = json.loads(original_freeze_raw)["files"]
    original_before = {name: sha(name) for name in original_files}
    if original_before != original_files:
        raise ValueError("the actual metadata freeze changed before testing")
    blocked = {"jax", "jaxlib", "numpy", "torch", "cupy", "flax", "optax", "scipy", "jaxtari", "exploration_platform"}
    if blocked & {name.split(".", 1)[0] for name in sys.modules}:
        raise AssertionError("a numerical module was already loaded")
    class NoNumericalImports(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split(".", 1)[0] in blocked:
                raise AssertionError("numerical import is forbidden in controller metadata tests")
    sys.meta_path.insert(0, NoNumericalImports())
    actions = []
    def audit(event, args):
        if event.startswith("socket.") and event != "socket.gethostname":
            raise AssertionError("network access is forbidden")
        if event == "ctypes.dlopen" or event == "ctypes.dlsym":
            raise AssertionError("real dynamic-library access is forbidden")
        if event == "open":
            name = str(args[0])
            if name.startswith(("/dev/nvidia", "/dev/dri")) or name == "/tmp/sl5nw_serval02_card1.lock":
                raise AssertionError("GPU device and standing-lock access are forbidden")
        if event == "subprocess.Popen":
            executable, command, cwd, env = args
            command = list(command)
            if not (len(command) == 4 and command[0] == sys.executable
                    and command[2] == "--prepared-unit"
                    and str(command[1]).startswith("/tmp/direct-controller-private-cpu-")
                    and str(command[1]).endswith("/campaign/data/private-cpu-fixture/code/measure_selected.py")
                    and str(command[3]) == str(Path(command[1]).parent.parent)
                    and str(cwd) == command[3]):
                raise AssertionError("only the exact private CPU fixture producer may be spawned")
            actions.append({"kind": "private_cpu_metadata_child", "command": command, "cwd": str(cwd)})
    sys.addaudithook(audit)
    spec = importlib.util.spec_from_file_location("direct_controller_regression", PACKET / "test_direct_qualification_controller.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if len(sys.argv) == 1:
        suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    else:
        if sys.argv[1:] != ["ControllerTests.test_late_launch_collision_preserves_existing_bytes_and_no_terminal"]:
            raise ValueError("only the exact corrected collision fixture may be selected separately")
        suite = unittest.defaultTestLoader.loadTestsFromName(sys.argv[1], module)
    started = time.monotonic()
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    source_after = {name: sha(PACKET / name) for name in SOURCE_NAMES}
    original_after = {name: sha(name) for name in original_files}
    if source_before != source_after or original_before != original_after:
        raise AssertionError("a tested or original frozen source changed during metadata testing")
    raw_stat = Path("/proc/self/stat").read_text()
    uid_line = "".join(line + "\n" for line in Path("/proc/self/status").read_text().splitlines() if line.startswith("Uid:"))
    step = os.environ["SLURM_STEP_ID"]
    record = {"schema_version": 1, "kind": "direct_qualification_controller_metadata_author_verification",
        "at_utc": datetime.now(timezone.utc).isoformat(), "job_id": "6783003", "step_id": step,
        "actual_hostname": socket.gethostname(), "actual_pid": os.getpid(),
        "actual_process_start_time": raw_stat[raw_stat.rfind(")") + 2:].split()[19],
        "actual_raw_stat": raw_stat, "actual_uid_line": uid_line,
        "actual_cgroup": Path("/proc/self/cgroup").read_text(),
        "actual_affinity": sorted(os.sched_getaffinity(0)), "source_before": source_before,
        "source_after": source_after, "original_inputs_before": original_before, "original_inputs_after": original_after,
        "original_packet_freeze_sha256": hashlib.sha256(original_freeze_raw).hexdigest(),
        "tests_run": result.testsRun, "passed": result.wasSuccessful(), "elapsed_seconds": time.monotonic() - started,
        "failures": [{"test": str(test), "traceback": text} for test, text in result.failures],
        "errors": [{"test": str(test), "traceback": text} for test, text in result.errors],
        "private_child_actions": actions, "private_cpu_fixtures": module.ControllerTests.fixture_results,
        "numerical_modules_loaded": sorted(blocked & {name.split(".", 1)[0] for name in sys.modules}),
        "actual_direct_contact": False, "actual_target_process_started": False, "gpu_or_standing_lock_access": False,
        "no_actual_gpu_qualification": True, "scientific_credit": 0, "case_credit": 0, "idea_credit": 0,
        "worker_admitted": False, "admission_granted": False}
    directory = PACKET / "controller_metadata_tests"
    directory.mkdir(exist_ok=True)
    path = directory / ("author_6783003_" + step + ".json")
    with path.open("xb") as stream:
        stream.write((json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    descriptor = os.open(directory, os.O_DIRECTORY)
    os.fsync(descriptor)
    os.close(descriptor)
    print(json.dumps({"receipt": str(path), "sha256": sha(path), "tests_run": result.testsRun,
                      "passed": result.wasSuccessful(), "step_id": step}, sort_keys=True))
    return 0 if result.wasSuccessful() else 1

if __name__ == "__main__":
    raise SystemExit(main())

