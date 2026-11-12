"""Can the Warp environment state be saved as NumPy arrays and restored, as the trainer's checkpoint does?

Run on a compute node (processors):  <platform_jax_warp python> diagnose_checkpoint_roundtrip.py
Steps 64 environments 20 times, writes every leaf of the state to NumPy arrays, restores them into the tree
of a fresh reset(), steps 20 more times, and compares with 40 uninterrupted steps. Also prints how many
leaves and bytes the state holds, for the Warp and for the JAX environment.
"""
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(BASE / "src"))
sys.path.insert(0, str(BASE / "benchmarks" / "env"))
import antmaze_warp_checks as checks  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from exploration_platform.envs.antmaze.am_common import preset  # noqa: E402
from exploration_platform.envs.antmaze.mjx_antmaze import JaxAntMaze  # noqa: E402
from exploration_platform.envs.antmaze.warp_antmaze import WarpAntMaze  # noqa: E402

N = 64
act = checks.torques("checkpoint torques", 40, N)
for name, env in (("warp", WarpAntMaze(preset("umaze"), N, 1)), ("jax", JaxAntMaze(preset("umaze"), N, 1))):
    call = checks.rollout_call(env)
    whole, _ = call(env.reset(), act)
    half, _ = call(env.reset(), act[:20])
    # GOAL: the checkpoint: every leaf as a NumPy array; the restore: the same arrays in a fresh tree.
    saved = [np.asarray(leaf) for leaf in jax.tree.leaves(half)]
    restored = jax.tree.unflatten(jax.tree.structure(env.reset()), [jnp.asarray(x) for x in saved])
    resumed, _ = call(restored, act[20:])
    same = all(np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True)
               for a, b in zip(jax.tree.leaves(whole.data.qpos), jax.tree.leaves(resumed.data.qpos)))
    worst = float(np.abs(np.asarray(whole.data.qpos) - np.asarray(resumed.data.qpos)).max())
    n_bytes = sum(x.nbytes for x in saved)
    print(f"{name}: {len(saved)} leaves, {n_bytes / N:,.0f} bytes per environment at {N} environments; positions after"
          f" save, restore and 20 more steps equal the uninterrupted run bit for bit: {same} (largest difference {worst:.2e});"
          f" step counts equal: {np.array_equal(np.asarray(whole.step_count), np.asarray(resumed.step_count))}")
