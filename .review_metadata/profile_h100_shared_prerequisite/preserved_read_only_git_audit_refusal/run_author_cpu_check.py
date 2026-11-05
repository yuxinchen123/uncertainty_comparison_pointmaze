"""Run the nonwriting real-plan metadata regression under a strict CPU audit."""
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import time
import unittest

DIRECTORY = Path(__file__).resolve().parent
ROOT = DIRECTORY.parents[1]
SCRIPTS = ROOT / "10_jax_exploration_platform/scripts/atari_pixel_campaign"
FORBIDDEN_MODULES = {"jax", "jaxlib", "numpy", "torch", "flax", "optax", "scipy", "cupy", "jaxtari", "exploration_platform"}
events = []


def audit(event, arguments):
    """Allow metadata and read-only source Git while refusing numerical, device and network work."""
    if event == "import" and arguments[0].split(".")[0] in FORBIDDEN_MODULES:
        raise AssertionError("CPU fixture refused numerical import: " + arguments[0])
    if event in {"ctypes.dlopen", "socket.connect", "socket.connect_ex", "os.system", "os.exec", "os.posix_spawn", "os.fork"}:
        raise AssertionError("CPU fixture refused operation: " + event)
    if event == "subprocess.Popen":
        command = arguments[1]
        if (not isinstance(command, (list, tuple)) or Path(command[0]).name != "git"
                or not any(value in command for value in ("rev-parse", "show", "ls-tree", "status", "cat-file"))):
            raise AssertionError("CPU fixture refused non-source subprocess")
        events.append(list(command))
    if event == "open":
        path = arguments[0]
        if isinstance(path, (str, bytes)):
            value = os.fsdecode(path)
            if value.startswith(("/dev/nvidia", "/dev/dri")):
                raise AssertionError("CPU fixture refused device read")
            if arguments[2] & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC):
                if not value.startswith(str(DIRECTORY) + "/"):
                    raise AssertionError("CPU fixture refused an output outside its review directory")


def main():
    """Keep actual source/plan hashes, test output and scope limits in one exclusive receipt."""
    if os.environ.get("SLURM_JOB_ID") != "6825983":
        raise AssertionError("run this check only in the current owned CPU allocation")
    source = SCRIPTS / "profile_selected.py"
    test = SCRIPTS / "tests/test_profile_h100_shared_prerequisite.py"
    before = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (source, test)}
    started = time.monotonic()
    # Standard-library ctypes initializes only its own-process handle before the audit; no library path is opened.
    import ctypes
    own_process_bootstrap = {"module": ctypes.__file__, "handle_name": ctypes.pythonapi._name, "before_audit_hook": True}
    if own_process_bootstrap["handle_name"] is not None:raise AssertionError("the stdlib own-process bootstrap differs")
    sys.addaudithook(audit)
    spec = importlib.util.spec_from_file_location("h100_shared_prerequisite_tests", test)
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(module))
    after = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (source, test)}
    if before != after:raise AssertionError("the held source changed during review")
    with (DIRECTORY / "author_test_output.raw").open("x") as output:output.write(stream.getvalue())
    receipt = {"kind": "author_h100_shared_dependency_actual_plan_metadata_check", "passed": result.wasSuccessful(),
        "owned_job_id": os.environ["SLURM_JOB_ID"], "actual_step_id": os.environ.get("SLURM_STEP_ID"),
        "methods": result.testsRun, "elapsed_seconds": time.monotonic() - started,
        "source_hashes_before": before, "source_hashes_after": after, "read_only_git_commands": events,
        "integration_scope": "Actual frozen H100 plan and complete staged verify_prepared gate; only the new selector is replaced in the isolated module namespace. No recorded plan, staged file, source/hash gate or upstream reference is changed.",
        "new_source_qualification_claimed": False, "actual_preparation_executed": False,
        "gpu_operations": 0, "numerical_operations": 0, "target_launches": 0, "scientific_credit": 0, "own_process_stdlib_bootstrap": own_process_bootstrap}
    with (DIRECTORY / "author_cpu_check.json").open("x") as output:json.dump(receipt, output, indent=2, sort_keys=True);output.write("\n")
    print(stream.getvalue())
    print(json.dumps({"passed": receipt["passed"], "methods": receipt["methods"], "elapsed_seconds": receipt["elapsed_seconds"]}))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
