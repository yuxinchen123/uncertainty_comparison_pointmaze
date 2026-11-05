"""Run only the new group metadata tests on an already-owned numeric CPU step."""
import ctypes
import importlib.abc
import importlib.util
import json
import os
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "10_jax_exploration_platform/scripts/atari_pixel_campaign"
NUMERICAL = {"jax", "jaxlib", "numpy", "torch", "cupy", "flax", "optax", "jaxtari", "scipy"}


class NoNumerical(importlib.abc.MetaPathFinder):
    """Refuse numerical package imports during synthetic metadata checks."""
    def find_spec(self, name, path=None, target=None):
        """Reject the package before Python imports it."""
        if name.split(".")[0] in NUMERICAL:
            raise RuntimeError("numerical import forbidden during group metadata tests")
        return None


def main():
    """Verify the owned allocation and run only the separate new test module."""
    assert os.environ.get("SLURM_JOB_ID") == "6783003"
    assert os.environ.get("SLURM_STEP_ID", "").isdecimal()
    assert os.getuid() == os.geteuid() == 2618919
    assert os.getresuid() == (2618919, 2618919, 2618919)
    sys.meta_path.insert(0, NoNumerical())
    ctypes.CDLL = ctypes.PyDLL = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("library load forbidden"))
    sys.addaudithook(lambda event, args: (_ for _ in ()).throw(RuntimeError("network forbidden"))
        if event in {"socket.connect", "socket.bind", "socket.getaddrinfo"} else None)
    path = SCRIPTS / "tests/test_configured_a4500_group.py"
    specification = importlib.util.spec_from_file_location("test_configured_a4500_group", path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(module))
    assert not (NUMERICAL & {name.split(".")[0] for name in sys.modules})
    print(json.dumps({"passed": result.wasSuccessful(), "methods": result.testsRun,
        "actual_job": os.environ["SLURM_JOB_ID"], "actual_step": os.environ["SLURM_STEP_ID"],
        "numerical_imports": 0, "gpu_actions_requested": 0, "scientific_credit": 0,
        "admission_granted": False}))
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
