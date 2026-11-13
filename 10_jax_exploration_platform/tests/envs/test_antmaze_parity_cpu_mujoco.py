"""MJX against C MuJoCo on the same AntMaze model, and the tuned solver against the reference.

Three gates (spec.md, "Correctness gates" 1 and 2):
  1. one-step parity, split by contact: from 50 states along a C-MuJoCo trajectory, one env
     step in MJX (float32) against one in C MuJoCo (float64). At contact-free states the two
     agree to float32 roundoff (the 2026-08-18 diagnostic measured median 3.9e-7, and MJX f64
     against C f64 agreed to 4.4e-16 — the same algorithm). At states inside a contact event
     they differ at the centimetre level, because MJX's collision functions differ from C's by
     documented design (different contact-point sets for the same geom pair); the gate bounds
     that difference rather than pretending it away.
  2. trajectory divergence structure: 20 env steps of shared random torques from the spawn
     agree to roundoff before the first contact, then diverge exponentially — two samples of
     the same chaotic dynamics, bounded here by one maze cell over the horizon. A
     whole-trajectory closeness bound would be pretending chaos away.
  3. equilibrium preservation: settled long enough (1,200 env steps — at 400 the tuned model
     is still moving at |qvel| 1.4e-2), BOTH integrators come to rest at exactly the same
     pose (z = 0.38248 m), and a state settled under either stays settled under the other to
     ~3e-11. The gate checks both directions.

Every gate runs twice: at the reference's time step (0.01 s, 5 physics steps per environment
step) and at the time step the AntMaze baseline runs with since 2026-11-12 (0.005 s, 10 physics
steps; spec.md, deviation 10), each against C MuJoCo on the model of the same configuration.

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
# the physics of the AntMaze baseline: half the time step, twice the physics steps (0.05 s per
# environment step, as with CFG)
STABLE = replace(CFG, physics_time_step=0.005, frame_skip=10)
CONFIGURATIONS = {"time step 0.01 s, 5 physics steps": CFG, "time step 0.005 s, 10 physics steps": STABLE}


def c_env_step(m, d, ctrl, frame_skip=CFG.frame_skip):
    """One reference env step in C MuJoCo: frame_skip physics steps at held ctrl.

    Returns the largest contact count seen at any of the substeps, so a caller can tell a
    genuinely contact-free window from one that touched a wall or the floor mid-window.
    """
    d.ctrl[:] = ctrl
    ncon_max = d.ncon
    for _ in range(frame_skip):
        mujoco.mj_step(m, d)
        ncon_max = max(ncon_max, d.ncon)
    return ncon_max


def c_trajectory(env, steps, key, cfg=CFG):
    """C-MuJoCo trajectory from the spawn under keyed random torques, on the model of `cfg`.

    before: nothing; after: qpos [steps+1, nq], qvel [steps+1, nv], acts [steps, nu],
    ncon [steps+1] (contacts at each stored state), all numpy
    """
    m = mujoco.MjModel.from_xml_string(build_antmaze_xml(cfg))
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    d.qpos[:] = np.asarray(env.init_qpos, np.float64)
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)
    qs, vs, acts, ncons = [d.qpos.copy()], [d.qvel.copy()], [], [d.ncon]
    for k in range(steps):
        a = np.asarray(jax.random.uniform(jax.random.fold_in(key, k), (m.nu,),
                                          jnp.float32, -1.0, 1.0), np.float64)
        c_env_step(m, d, a, cfg.frame_skip)
        qs.append(d.qpos.copy())
        vs.append(d.qvel.copy())
        acts.append(a)
        ncons.append(d.ncon)
    return np.stack(qs), np.stack(vs), np.stack(acts), np.asarray(ncons), m


def check_one_step_parity(cfg):
    """From states along a C trajectory, MJX matches C tightly away from contacts, and within
    the documented collision-function difference at them."""
    env = JaxAntMaze(cfg, 1, 1)
    qs, vs, acts, ncons, m = c_trajectory(env, 50, jax.random.PRNGKey(11), cfg)
    d = mujoco.MjData(m)
    step = jax.jit(env.step)
    free, contact = [], []
    for k in range(50):
        # C MuJoCo one step from state k, remembering whether ANY substep saw a contact
        mujoco.mj_resetData(m, d)
        d.qpos[:], d.qvel[:] = qs[k], vs[k]
        mujoco.mj_forward(m, d)
        ncon_in_window = c_env_step(m, d, acts[k], cfg.frame_skip)
        # MJX one step from the same state
        s = env.reset()
        s = s._replace(data=s.data.replace(qpos=jnp.asarray(qs[k], jnp.float32)[None],
                                           qvel=jnp.asarray(vs[k], jnp.float32)[None]))
        s2, *_ = step(s, jnp.asarray(acts[k], jnp.float32).reshape(1, 1, 8))
        err = float(np.abs(np.asarray(s2.data.qpos[0]) - d.qpos).max())
        # contact-free means no substep of the window saw a contact
        (free if ncon_in_window == 0 else contact).append(err)
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
    print("ok one-step parity")


def check_trajectory_divergence_structure(cfg):
    """From the spawn, MJX tracks C to roundoff until the first contact, then the two diverge
    exponentially as chaos amplifies the collision-function difference; the gate checks the
    pre-contact agreement and that 20 steps of divergence stay under one maze cell."""
    env = JaxAntMaze(cfg, 1, 1)
    qs, vs, acts, ncons, m = c_trajectory(env, 20, jax.random.PRNGKey(5), cfg)
    s = env.reset()
    step = jax.jit(env.step)
    errs = []
    for k in range(20):
        s, *_ = step(s, jnp.asarray(acts[k], jnp.float32).reshape(1, 1, 8))
        errs.append(float(np.abs(np.asarray(s.data.qpos[0]) - qs[k + 1]).max()))
    # the first stored state with a contact, from the C trajectory's own contact counts
    first_contact = int(np.argmax(ncons > 0)) if (ncons > 0).any() else len(ncons)
    pre = errs[:max(first_contact - 1, 1)]
    print(f"20-step trajectory: first contact at state {first_contact}, pre-contact worst "
          f"{max(pre):.2e}, final divergence {errs[-1]:.2e}")
    assert max(pre) < 1e-4, ("disagreement before any contact is beyond roundoff: "
                             f"{max(pre):.2e}")
    assert max(errs) < 4.0, ("20-step divergence beyond one maze cell suggests different "
                             f"physics, not chaos: {max(errs):.2e}")
    assert all(np.isfinite(errs)), errs
    print("ok trajectory divergence structure")


def check_equilibrium_preservation(tuned):
    """Settled long enough, the tuned solver rests at exactly the reference's pose, and a
    state settled under either integrator stays settled under the other. The reference is RK4 at
    0.01 s with MuJoCo's default iteration counts, whatever the tuned configuration's time step."""
    env = JaxAntMaze(tuned, 1, 1)
    reference = replace(CFG, integrator="RK4", solver_iterations=100, ls_iterations=50)

    def settle(cfg, qpos=None, qvel=None, steps=1200):
        """Run `steps` passive env steps from the spawn (or a given state); returns (qpos, qvel).

        1,200 steps because the transient is slow: at 400 the tuned model still moves at
        |qvel| about 1.4e-2 and sits 0.18 m above where it finally rests.
        """
        m = mujoco.MjModel.from_xml_string(build_antmaze_xml(cfg))
        d = mujoco.MjData(m)
        mujoco.mj_resetData(m, d)
        d.qpos[:] = np.asarray(env.init_qpos, np.float64) if qpos is None else qpos
        d.qvel[:] = 0.0 if qvel is None else qvel
        for _ in range(steps * cfg.frame_skip):
            mujoco.mj_step(m, d)
        return d.qpos.copy(), d.qvel.copy()

    heights = {}
    for a, b, tag in [(reference, tuned, "reference -> tuned"),
                      (tuned, reference, "tuned -> reference")]:
        qa, va = settle(a)
        qb, vb = settle(b, qa, va, steps=200)
        drift = np.abs(qb - qa).max()
        heights[tag] = (qa[2], qb[2])
        print(f"{tag}: settled z {qa[2]:.5f}, after handover z {qb[2]:.5f}, "
              f"max |dqpos| {drift:.2e}")
        assert drift < 1e-6, (tag, drift)
    # both integrators rest at the same pose, not merely each at "a" pose
    z_ref, z_tuned = heights["reference -> tuned"][0], heights["tuned -> reference"][0]
    assert abs(z_ref - z_tuned) < 1e-4, (z_ref, z_tuned)
    print("ok equilibrium preservation")


def test_one_step_parity():
    """Gate 1 at the reference's time step."""
    check_one_step_parity(CFG)


def test_trajectory_divergence_structure():
    """Gate 2 at the reference's time step."""
    check_trajectory_divergence_structure(CFG)


def test_equilibrium_preservation():
    """Gate 3 at the reference's time step."""
    check_equilibrium_preservation(CFG)


def test_one_step_parity_at_the_stable_time_step():
    """Gate 1 at 0.005 s with 10 physics steps per environment step."""
    check_one_step_parity(STABLE)


def test_trajectory_divergence_structure_at_the_stable_time_step():
    """Gate 2 at 0.005 s with 10 physics steps per environment step."""
    check_trajectory_divergence_structure(STABLE)


def test_equilibrium_preservation_at_the_stable_time_step():
    """Gate 3 at 0.005 s with 10 physics steps per environment step."""
    check_equilibrium_preservation(STABLE)


def test_the_time_step_field_sets_the_models_time_step_and_nothing_else():
    """The default configuration builds the model of every earlier run (0.01 s); 0.005 changes the
    option line's time step only; a time step of zero or a frame skip of zero is refused."""
    default_xml, stable_xml = build_antmaze_xml(CFG), build_antmaze_xml(STABLE)
    assert 'timestep="0.01"' in default_xml and 'timestep="0.005"' in stable_xml
    assert default_xml.replace('timestep="0.01"', 'timestep="0.005"') == stable_xml
    assert mujoco.MjModel.from_xml_string(default_xml).opt.timestep == 0.01
    assert mujoco.MjModel.from_xml_string(stable_xml).opt.timestep == 0.005
    assert CFG.physics_time_step == 0.01 and CFG.frame_skip == 5
    # one environment step lasts 0.05 s in both
    assert abs(STABLE.physics_time_step * STABLE.frame_skip - CFG.physics_time_step * CFG.frame_skip) < 1e-12
    for bad in (replace(CFG, physics_time_step=0.0), replace(CFG, physics_time_step=-0.01),
                replace(CFG, frame_skip=0)):
        try:
            build_antmaze_xml(bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad.physics_time_step} s with frame skip {bad.frame_skip}")


if __name__ == "__main__":
    for name, cfg in CONFIGURATIONS.items():
        print(name)
        check_one_step_parity(cfg)
        check_trajectory_divergence_structure(cfg)
        check_equilibrium_preservation(cfg)
    test_the_time_step_field_sets_the_models_time_step_and_nothing_else()
