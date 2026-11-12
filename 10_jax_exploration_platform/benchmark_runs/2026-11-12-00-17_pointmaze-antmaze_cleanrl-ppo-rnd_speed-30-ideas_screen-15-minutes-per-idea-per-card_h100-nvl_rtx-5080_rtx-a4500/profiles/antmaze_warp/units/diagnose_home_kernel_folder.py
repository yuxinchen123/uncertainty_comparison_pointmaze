"""Does Warp's default kernel folder (the home folder's ~/.cache/warp/<version>) work on a compute node?

Run on a compute node with WARP_CACHE_PATH unset:  <platform_jax_warp python> diagnose_home_kernel_folder.py
Builds the Warp AntMaze environment at 4 environments, steps it 10 times, and prints where the kernels went.
"""
import os
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(BASE / "src"))
sys.path.insert(0, str(BASE / "benchmarks" / "env"))
assert "WARP_CACHE_PATH" not in os.environ, "unset WARP_CACHE_PATH for this diagnostic"
import antmaze_warp_checks as checks  # noqa: E402
import warp as wp  # noqa: E402
from exploration_platform.envs.antmaze.am_common import preset  # noqa: E402
from exploration_platform.envs.antmaze.warp_antmaze import WarpAntMaze  # noqa: E402

env = WarpAntMaze(preset("umaze"), 4, 1)
t0 = time.perf_counter()
state, _ = checks.rollout_call(env)(env.reset(), checks.torques("contract torques", 10, 4))
folder = Path(wp.config.kernel_cache_dir)
print(f"first call {time.perf_counter() - t0:.1f} s; kernel folder {folder}; {sum(1 for p in folder.iterdir() if p.is_dir())}"
      f" kernel folders; pool {env.overflow_report(state)}")
