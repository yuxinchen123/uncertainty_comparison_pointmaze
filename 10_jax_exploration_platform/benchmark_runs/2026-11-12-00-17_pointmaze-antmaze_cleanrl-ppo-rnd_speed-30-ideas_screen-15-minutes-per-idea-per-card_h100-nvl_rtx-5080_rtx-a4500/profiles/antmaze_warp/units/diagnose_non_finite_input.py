"""What the Warp and the JAX environment make of a non-finite number put into one coordinate of the state.

Run on a compute node (processors):  <platform_jax_warp python> diagnose_non_finite_input.py
Prints, per poisoned coordinate, the state of the poisoned environment after one environment step in both
implementations, before the environment's own respawn rule is applied (the physics step alone).
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
from mujoco import mjx  # noqa: E402
from exploration_platform.envs.antmaze.am_common import preset  # noqa: E402
from exploration_platform.envs.antmaze.mjx_antmaze import JaxAntMaze  # noqa: E402
from exploration_platform.envs.antmaze.warp_antmaze import WarpAntMaze  # noqa: E402

np.set_printoptions(precision=4, suppress=True, linewidth=200)
POISONS = (("qpos", 0, "torso x"), ("qpos", 3, "torso quaternion w"), ("qpos", 8, "a hinge angle"),
           ("qvel", 0, "torso x velocity"), ("qvel", 4, "torso angular velocity"), ("qvel", 8, "a hinge velocity"))
for name, env in (("warp", WarpAntMaze(preset("umaze"), 1, 2)), ("jax", JaxAntMaze(preset("umaze"), 1, 2))):
    def physics(data):
        """Five physics steps with zero torque, without the environment's reward and reset."""
        with jax.enable_x64(False):
            for _ in range(5):
                data = jax.vmap(mjx.step, in_axes=(None, 0))(env.model, data)
            return data
    physics = jax.jit(physics)
    for field, index, label in POISONS:
        for value in (np.nan, np.inf):
            state = env.reset()
            array = np.asarray(getattr(state.data, field)).copy()
            array[0, index] = value
            data = physics(state.data.replace(**{field: jnp.asarray(array)}))
            qpos, qvel = np.asarray(data.qpos), np.asarray(data.qvel)
            print(f"{name}: {value} in {label}: poisoned environment finite qpos {np.isfinite(qpos[0]).all()} qvel"
                  f" {np.isfinite(qvel[0]).all()}, quaternion norm {np.linalg.norm(qpos[0, 3:7]):.4f}, largest |qpos|"
                  f" {np.nanmax(np.abs(qpos[0])):.3g}, largest |qvel| {np.nanmax(np.abs(qvel[0])):.3g}; the other"
                  f" environment finite {np.isfinite(qpos[1]).all() and np.isfinite(qvel[1]).all()}, its z {qpos[1, 2]:.5f}")
            if name == "warp":
                print("   qpos", qpos[0], "\n   qvel", qvel[0])
