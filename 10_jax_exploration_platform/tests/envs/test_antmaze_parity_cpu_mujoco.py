"""MJX against C MuJoCo on the same AntMaze model, and the tuned solver against the reference.

Three gates (spec.md, "Correctness gates" 1 and 2):
  1. one-step parity, split by contact: from 50 states along a C-MuJoCo trajectory, one env
     step in MJX (float32) against one in C MuJoCo (float64). At contact-free states the two
     agree to float32 roundoff (the 2026-08-18 diagnostic measured median 3.9e-7, and MJX f64
     against C f64 agreed to 4.4e-16 — the same algorithm). At states inside a contact event
     they differ at the centimetre level, because MJX's collision functions differ from C's by
     documented design (different contact-point sets for the same geom pair); the gate bounds
     that difference rather than pretending it away.
  2. short-horizon trajectory agreement: 20 env steps of shared random torques from the spawn.
  3. settling equivalence: the tuned integrator/solver settings come to rest at the reference
     RK4 full-solver height on C MuJoCo (the deviation does not change the physics's answer).

Run: PYTHONNOUSERSITE=1 JAX_PLATFORMS=cpu <jax python> test_antmaze_parity_cpu_mujoco.py
"""
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
import exploration_platform  # noqa: E402,F401
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from exploration_platform.envs.antmaze.am_common import build_antmaze_xml, preset  # noqa: E402
from exploration_platform.envs.antmaze.mjx_antmaze import JaxAntMaze  # noqa: E402

CFG = preset("umaze")


def c_env_step(m, d, ctrl):
    """One reference env step in C MuJoCo: 5 physics steps at held ctrl."""
    d.ctrl[:] = ctrl
    for _ in range(CFG.frame_skip):
        mujoco.mj_step(m, d)


def c_trajectory(env, steps, key):
    """C-MuJoCo trajectory from the spawn under keyed random torques.

    before: nothing; after: qpos [steps+1, nq], qvel [steps+1, nv], acts [steps, nu],
    ncon [steps+1] (contacts at each stored state), all numpy
    """
    m = mujoco.MjModel.from_xml_string(build_antmaze_xml(CFG))
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    d.qpos[:] = np.asarray(env.init_qpos, np.float64)
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)
    qs, vs, acts, ncons = [d.qpos.copy()], [d.qvel.copy()], [], [d.ncon]
    for k in range(steps):
        a = np.asarray(jax.random.uniform(jax.random.fold_in(key, k), (m.nu,),
                                          jnp.float32, -1.0, 1.0), np.float64)
        c_env_step(m, d, a)
        qs.append(d.qpos.copy())
        vs.append(d.qvel.copy())
        acts.append(a)
        ncons.append(d.ncon)
    return np.stack(qs), np.stack(vs), np.stack(acts), np.asarray(ncons), m


def test_one_step_parity():
    """From states along a C trajectory, MJX matches C tightly away from contacts, and within
    the documented collision-function difference at them."""
    env = JaxAntMaze(CFG, 1, 1)
    qs, vs, acts, ncons, m = c_trajectory(env, 50, jax.random.PRNGKey(11))
    d = mujoco.MjData(m)
    step = jax.jit(env.step)
    free, contact = [], []
    for k in range(50):
        # C MuJoCo one step from state k
        mujoco.mj_resetData(m, d)
        d.qpos[:], d.qvel[:] = qs[k], vs[k]
        c_env_step(m, d, acts[k])
        # MJX one step from the same state
        s = env.reset()
        s = s._replace(data=s.data.replace(qpos=jnp.asarray(qs[k], jnp.float32)[None],
                                           qvel=jnp.asarray(vs[k], jnp.float32)[None]))
        s2, *_ = step(s, jnp.asarray(acts[k], jnp.float32).reshape(1, 1, 8))
        err = float(np.abs(np.asarray(s2.data.qpos[0]) - d.qpos).max())
        # contact-free means no contact at the start state and none after the reference step
        (free if ncons[k] == 0 and d.ncon == 0 else contact).append(err)
    print(f"one-step parity: {len(free)} contact-free states, worst |dqpos| "
          f"{max(free):.2e}; {len(contact)} states in contact, worst {max(contact):.2e}"
          if contact else f"one-step parity: all {len(free)} states contact-free, "
                          f"worst {max(free):.2e}")
    assert free, "the trajectory produced no contact-free state; the split test needs both"
    assert max(free) < 1e-4, ("contact-free one-step disagreement beyond float32 roundoff: "
                              f"{max(free):.2e}")
    if contact:
        assert max(contact) < 5e-2, ("one-step disagreement at contact events beyond the "
                                     f"documented collision-function difference: "
                                     f"{max(contact):.2e}")
    print("ok test_one_step_parity")


def test_short_horizon_trajectory():
    """20 shared-torque env steps from the spawn stay close between MJX f32 and C f64."""
    env = JaxAntMaze(CFG, 1, 1)
    qs, vs, acts, ncons, m = c_trajectory(env, 20, jax.random.PRNGKey(5))
    s = env.reset()
    step = jax.jit(env.step)
    errs = []
    for k in range(20):
        s, *_ = step(s, jnp.asarray(acts[k], jnp.float32).reshape(1, 1, 8))
        errs.append(float(np.abs(np.asarray(s.data.qpos[0]) - qs[k + 1]).max()))
    print(f"20-step trajectory: final |dqpos| = {errs[-1]:.2e}, max = {max(errs):.2e}")
    assert max(errs) < 5e-2, errs
    print("ok test_short_horizon_trajectory")


def test_settling_equivalence():
    """Tuned (implicitfast, it4/ls8) and reference (RK4, full solver) rest at the same height."""
    env = JaxAntMaze(CFG, 1, 1)
    heights = {}
    for tag, cfg in [("tuned", CFG),
                     ("reference", replace(CFG, integrator="RK4",
                                           solver_iterations=100, ls_iterations=50))]:
        m = mujoco.MjModel.from_xml_string(build_antmaze_xml(cfg))
        d = mujoco.MjData(m)
        mujoco.mj_resetData(m, d)
        # settle from the spawn cell — the map's origin is a wall cell on the umaze
        d.qpos[:] = np.asarray(env.init_qpos, np.float64)
        d.qvel[:] = 0.0
        for _ in range(400 * cfg.frame_skip):
            mujoco.mj_step(m, d)
        heights[tag] = d.qpos[2]
    print(f"settled z: tuned {heights['tuned']:.5f}, reference {heights['reference']:.5f}")
    assert abs(heights["tuned"] - heights["reference"]) < 1e-3, heights
    print("ok test_settling_equivalence")


if __name__ == "__main__":
    test_one_step_parity()
    test_short_horizon_trajectory()
    test_settling_equivalence()
