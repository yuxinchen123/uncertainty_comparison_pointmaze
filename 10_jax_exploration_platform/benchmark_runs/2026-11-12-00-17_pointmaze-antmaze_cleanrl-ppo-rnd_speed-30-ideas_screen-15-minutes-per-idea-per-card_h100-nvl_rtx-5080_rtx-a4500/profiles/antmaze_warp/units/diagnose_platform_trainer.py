"""The platform's own training iteration with the Warp environment, at a tiny shape on processors.

Run on a compute node:  <platform_jax_warp python> diagnose_platform_trainer.py
"""
import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(BASE / "benchmarks" / "env"))
import antmaze_warp_checks as checks  # noqa: E402

for implementation in ("warp", "jax"):
    print(json.dumps(checks.platform_training_iterations(implementation, 2, 2, num_steps=8, timed_iterations=2,
                                                         episode_steps=6)), flush=True)
