"""Run exact-source metadata fixtures in one owned CPU step without numerical packages."""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "10_jax_exploration_platform/scripts/atari_pixel_campaign"
FILES = [SCRIPTS / name for name in ("direct_historical_consumer.py", "direct_common_confirmation.py",
    "direct_measurement.py", "measure_selected.py", "tests/test_direct_historical_consumer.py",
    "tests/test_direct_common_confirmation.py", "tests/test_configured_measurement.py")]
READ_ONLY_PROBE = None
COLD_MAIN_PROBE = None
NUMERICAL = {"jax", "jaxlib", "numpy", "torch", "flax", "optax", "scipy", "cupy", "nvidia", "jaxtari",
    "exploration_platform", "jax_cuda12_plugin", "jax_cuda13_plugin", "tensorflow", "numba", "ml_dtypes"}


def digest(path):
    """Bind only the small exact Python source files used by this metadata fixture."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(event, values):
    """Refuse numerical imports, device access and all network or later ctypes operations."""
    if event == "import" and values[0].split(".")[0] in NUMERICAL:
        raise RuntimeError("numerical import forbidden in metadata fixture")
    if event in {"ctypes.dlopen", "ctypes.dlsym", "socket.connect", "socket.bind", "socket.getaddrinfo", "os.system"}:
        raise RuntimeError("library or network operation forbidden in metadata fixture")
    if event == "open" and isinstance(values[0], (str, bytes)) and os.fsdecode(values[0]).startswith(("/dev/nvidia", "/dev/dri")):
        raise RuntimeError("device operation forbidden in metadata fixture")
    if event == "subprocess.Popen":
        command = values[1]
        expected = ["/usr/bin/python3", "-I", "-S", "-B", str(SCRIPTS / "direct_historical_consumer.py"), "--read-only-stdin"]
        if command == expected:
            return
        if command[:5] == ["/usr/bin/python3", "-I", "-S", "-B", "-c"]:
            if len(command) == 7 and command[5] == COLD_MAIN_PROBE and command[6] == str(SCRIPTS / "direct_common_confirmation.py"):
                return
            if len(command) == 8 and command[5] == READ_ONLY_PROBE:
                import measure_selected as selected
                names = json.loads(command[7])
                rosters = [list(selected.required_helpers(False, True)), list(selected.required_helpers(False, True, False, True)),
                    list(selected.required_helpers(False, True, True, True))]
                folder = Path(command[6])
                if names in rosters and folder.resolve(strict=True) == folder and folder.as_posix().startswith("/tmp/") \
                        and all(digest(folder / name) == digest(SCRIPTS / name) for name in names):
                    return
        raise RuntimeError("unlisted process forbidden in metadata fixture")


def main():
    """Capture exact source and actual owned process before and after these synthetic checks."""
    if os.environ.get("SLURM_JOB_ID") != "6783003" or not os.environ.get("SLURM_STEP_ID", "").isdigit():
        raise RuntimeError("only the exact owned CPU allocation may run these fixtures")
    before = {str(path): digest(path) for path in FILES}
    sys.path.insert(0, str(SCRIPTS))
    sys.path.insert(0, str(SCRIPTS / "tests"))
    import direct_common_confirmation as common
    bootstrap = common.stdlib_ctypes_bootstrap()
    common.validate_bootstrap_record(bootstrap)
    global READ_ONLY_PROBE, COLD_MAIN_PROBE
    from test_configured_measurement import ISOLATED_IMPORT_PROBE
    READ_ONLY_PROBE = ISOLATED_IMPORT_PROBE
    tree = ast.parse((SCRIPTS / "tests/test_direct_common_confirmation.py").read_text())
    method = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
        and node.name == "test_fresh_interpreter_real_main_import_order_bootstraps_only_stdlib_then_refuses_later_ctypes")
    COLD_MAIN_PROBE = next(node.value.value for node in ast.walk(method) if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "probe" for target in node.targets))
    sys.addaudithook(audit)
    suite = unittest.TestSuite()
    for index, filename in enumerate(("test_direct_historical_consumer.py", "test_direct_common_confirmation.py")):
        path = SCRIPTS / "tests" / filename
        specification = importlib.util.spec_from_file_location("historical_metadata_tests_" + str(index), path)
        module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(module)
        suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(module))
    started = time.time()
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    after = {str(path): digest(path) for path in FILES}
    if before != after or NUMERICAL.intersection(sys.modules):
        raise RuntimeError("test source changed or a numerical module was imported")
    import direct_historical_consumer as historical
    receipt = {"schema_version": 1, "kind": "synthetic_historical_consumer_focused_metadata_checks",
        "passed": result.wasSuccessful(), "tests_run": result.testsRun,
        "failures": len(result.failures), "errors": len(result.errors), "source_sha256": before,
        "job_id": os.environ["SLURM_JOB_ID"], "actual_step_id": os.environ["SLURM_STEP_ID"],
        "actual_process": historical.process(os.getpid()), "started_at_seconds": started,
        "finished_at_seconds": time.time(), "stdlib_ctypes_bootstrap": bootstrap,
        "numerical_imports": [], "gpu_operations": 0, "scientific_credit": 0,
        "historical_saved_evidence_verified": False, "admission_granted": False,
        "production_child_time_limit_seconds": historical.WAIT_SECONDS, "digest_cache": "disabled"}
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
