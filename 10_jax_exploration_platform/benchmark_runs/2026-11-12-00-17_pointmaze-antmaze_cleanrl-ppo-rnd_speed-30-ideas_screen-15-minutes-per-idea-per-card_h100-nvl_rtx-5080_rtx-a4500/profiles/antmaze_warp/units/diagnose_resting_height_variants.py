"""The resting height of the Warp ant with fewer solver or line-search iterations, on processors.

Run on a compute node:  <platform_jax_warp python> diagnose_resting_height_variants.py
"""
import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(BASE / "benchmarks" / "env"))
import antmaze_warp_checks as checks  # noqa: E402

for changes in ({"solver_iterations": 2}, {"solver_iterations": 1}, {"ls_iterations": 4},
                {"solver_iterations": 2, "ls_iterations": 4}):
    print(changes, json.dumps(checks.resting_height(changes=changes, with_jax=False)), flush=True)
