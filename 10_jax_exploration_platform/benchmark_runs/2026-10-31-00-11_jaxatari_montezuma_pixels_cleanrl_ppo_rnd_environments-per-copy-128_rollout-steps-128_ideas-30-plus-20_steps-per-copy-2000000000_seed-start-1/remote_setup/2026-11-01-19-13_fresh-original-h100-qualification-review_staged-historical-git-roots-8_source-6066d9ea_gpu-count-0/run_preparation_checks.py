"""Run source-only controls on the exact existing owned CPU monitor step."""
import ctypes
import importlib.abc
import importlib.util
import json
import os
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).absolute().parent
NUMERICAL = {"jax", "jaxlib", "numpy", "torch", "cupy", "flax", "optax", "jaxtari", "scipy"}


class NoNumerical(importlib.abc.MetaPathFinder):
    """Refuse numerical imports in this metadata-only source test."""
    def find_spec(self, name, path=None, target=None):
        """Refuse the named numerical package before it is imported."""
        if name.split(".")[0] in NUMERICAL:
            raise RuntimeError("numerical imports are forbidden in source preparation")
        return None


def main():
    """Check actual step ownership and run the controlled source tests."""
    assert os.environ.get("SLURM_JOB_ID") == "6783003"
    assert os.environ.get("SLURM_STEP_ID", "").isdecimal()
    assert os.getuid() == os.geteuid() == 2618919
    assert os.getresuid() == (2618919, 2618919, 2618919)
    sys.meta_path.insert(0, NoNumerical())
    ctypes.CDLL = ctypes.PyDLL = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("library loads forbidden"))
    sys.addaudithook(lambda event, args: (_ for _ in ()).throw(RuntimeError("network forbidden"))
        if event in {"socket.connect", "socket.bind", "socket.getaddrinfo"} else None)
    spec = importlib.util.spec_from_file_location("test_fresh_direct_raw_review", ROOT / "test_fresh_direct_raw_review.py")
    tests = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tests)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(tests))
    assert not (NUMERICAL & {name.split(".")[0] for name in sys.modules})
    print(json.dumps({"passed": result.wasSuccessful(), "test_methods": result.testsRun,
        "prior_source_and_synthetic_control_cases": 23, "mapped_historical_git_roots": 8,
        "mapped_root_controls_run": True, "actual_result_read": False,
        "actual_job": os.environ["SLURM_JOB_ID"], "actual_step": os.environ["SLURM_STEP_ID"],
        "numerical_imports": 0, "gpu_actions": 0, "scheduler_queries": 0,
        "scientific_credit": 0, "admission_granted": False}))
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
