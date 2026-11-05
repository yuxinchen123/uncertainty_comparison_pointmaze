"""Run only owned-CPU group metadata fixtures and preserve complete source/return evidence."""
import ctypes
import datetime
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import unittest


def digest(path):
    """Hash an exact source file without loading its contents as native code."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    """Require this campaign's owned CPU step and forbid numerical, scheduler, GPU and network work."""
    freeze, receipt = map(Path, sys.argv[1:3])
    if os.environ.get("SLURM_JOB_ID") != "6783003" or not os.environ.get("SLURM_STEP_ID", "").isdigit():
        raise ValueError("run only inside a numeric step of owned CPU monitor6783003")
    if receipt.exists() or os.getuid() != os.geteuid() or os.getuid() != 2618919:
        raise ValueError("retain previous receipts and require this campaign owner")
    expected = json.loads(freeze.read_text())["files"]
    before = {path: digest(path) for path in expected}
    if before != expected:
        raise ValueError("frozen group source changed before testing")
    scripts = Path(__file__).resolve().parents[2] / "10_jax_exploration_platform/scripts/atari_pixel_campaign"
    allowed_python = str(Path(sys.executable).absolute())

    def audit(event, args):
        """Allow only source-fixture Git and isolated Python metadata children after the declared stdlib bootstrap."""
        if event == "subprocess.Popen":
            executable, command = args[:2]
            if not isinstance(command, (list, tuple)) or not command or type(command[0]) is not str:
                raise RuntimeError("literal metadata command required")
            if command[0] == "git":
                if executable != "git":
                    raise RuntimeError("fixture Git executable substitution forbidden")
            elif command[0] == allowed_python:
                if executable != allowed_python or list(command[1:4]) != ["-I", "-S", "-B"]:
                    raise RuntimeError("only the exact isolated metadata Python child is allowed")
            else:
                raise RuntimeError("scheduler, tool and other subprocess operations are forbidden")
        if event in {"os.system", "os.posix_spawn", "socket.connect", "socket.bind", "socket.getaddrinfo", "ctypes.dlopen", "ctypes.dlsym"}:
            raise RuntimeError("network, command shell, device or later library operations forbidden")
        if event == "import" and str(args[0]).split(".")[0] in {
                "jax", "jaxlib", "numpy", "torch", "jaxtari", "cupy", "pynvml", "flax", "optax", "scipy"}:
            raise RuntimeError("numerical imports forbidden in group metadata fixtures")

    sys.addaudithook(audit)
    sys.path.insert(0, str(scripts)); sys.path.insert(0, str(scripts / "tests"))
    modules = []
    for name in ("test_configured_a4500_group", "test_group_runtime", "test_group_runtime_predicate_corrections"):
        specification = importlib.util.spec_from_file_location(name, scripts / "tests" / (name + ".py"))
        module = importlib.util.module_from_spec(specification)
        sys.modules[name] = module
        specification.loader.exec_module(module)
        modules.append(module)
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromModule(module) for module in modules)
    def names(group):
        """Retain distinct actual test identities rather than counting printed lines."""
        if isinstance(group, unittest.TestSuite):
            return [name for item in group for name in names(item)]
        return [group.id()]
    identities = names(suite)
    output = io.StringIO()
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    result = unittest.TextTestRunner(stream=output, verbosity=2).run(suite)
    after = {path: digest(path) for path in expected}
    record = {"kind": "group_runtime_metadata_tests", "passed": result.wasSuccessful() and before == after,
        "source_freeze": {"path": str(freeze), "sha256": digest(freeze)}, "source_before": before, "source_after": after,
        "tests_run": result.testsRun, "test_ids": identities, "unique_test_ids": len(set(identities)),
        "failures": len(result.failures), "errors": len(result.errors), "test_output": output.getvalue(),
        "job_id": os.environ["SLURM_JOB_ID"], "step_id": os.environ["SLURM_STEP_ID"], "pid": os.getpid(),
        "started_at_utc": started, "finished_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "stdlib_bootstrap_before_audit": "ctypes imports its own-process Python API before the audit hook; all later library operations refused",
        "actual_scheduler_or_gpu_or_tool_operations": 0, "scientific_credit": 0}
    with receipt.open("x") as stream:
        json.dump(record, stream, indent=2, sort_keys=True); stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    print(json.dumps({"passed": record["passed"], "tests_run": result.testsRun, "errors": record["errors"],
        "failures": record["failures"], "step": record["step_id"], "receipt_sha256": digest(receipt)}, sort_keys=True))
    if not record["passed"]:
        print(output.getvalue())
    return 0 if record["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
