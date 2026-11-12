"""The two speed options of the MJX AntMaze stepper against the stepper as shipped.

`AntMazeConfig.solver_loop="fixed_count"` runs MJX's Newton passes in a loop of fixed length
instead of `jax.lax.while_loop`; `AntMazeConfig.carried_state="next_step_inputs"` carries only
positions, velocities, the solver's start values and the clock between steps. Both must leave the
physics as it is:

  1. the defaults are the stepper as shipped, and a wrong value is refused;
  2. the fixed-count loop is the same algorithm as `mjx.step` and differs from it by rounding
     only. The two are compiled as different programs, so they are not bitwise equal: measured
     2026-11-12 on processors, one float32 physics step from the same state differs by at most
     7.9e-6 relative (50 states, 36 with contacts). The gate is therefore in two parts: in
     float32, one step from 50 states stays within 1e-4 relative; in float64, where rounding is
     nine digits smaller, one step stays within 1e-10 and 50 steps run freely within 1e-8;
  3. carrying only the next step's inputs is bitwise equal to carrying the whole data for 100
     environment steps, through a truncation and its reset, with either solver loop;
  4. the non-finite respawn works on the carried arrays;
  5. the fused trainer composes and trains with both options on.

Run: PYTHONNOUSERSITE=1 JAX_PLATFORMS=cpu <jax python> test_antmaze_mjx_options.py
"""
import functools
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
import exploration_platform  # noqa: E402,F401  (sets jax_enable_x64=True, the platform default)
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from mujoco import mjx  # noqa: E402
from exploration_platform.agents.ppo.config import PPOConfig  # noqa: E402
from exploration_platform.envs.antmaze.am_common import build_antmaze_xml, preset  # noqa: E402
from exploration_platform.envs.antmaze.mjx_antmaze import (  # noqa: E402
    CarriedPhysics, JaxAntMaze, physics_step_fixed_count)
from exploration_platform.training.runner import Runner  # noqa: E402

BOTH = dict(solver_loop="fixed_count", carried_state="next_step_inputs")
CARRIED_NAMES = ("qpos", "qvel", "qacc_warmstart", "time")


def torques(key, k, env):
    """Uniform torques in [-1, 1] for step k, shape [C, N, 8], whatever the options of the env."""
    return jax.random.uniform(jax.random.fold_in(key, k), (env.C, env.N, env.act_dim),
                              jnp.float32, -1.0, 1.0)


def rollout(env, key, steps):
    """Step random torques from the spawn; per step: the five outputs, carried arrays, counters."""
    step, state, trace = jax.jit(env.step), env.reset(), []
    for k in range(steps):
        state, *outputs = step(state, torques(key, k, env))
        carried = [np.asarray(getattr(state.data, name)) for name in CARRIED_NAMES]
        counters = [np.asarray(state.step_count), np.asarray(state.reset_count),
                    np.asarray(state.nan_count)]
        trace.append([np.asarray(x) for x in outputs] + carried + counters)
    return trace


def test_defaults_are_the_stepper_as_shipped():
    """The default configuration calls mjx.step on the whole data; wrong values are refused."""
    cfg = preset("umaze")
    assert (cfg.solver_loop, cfg.carried_state) == ("while_loop", "whole_data")
    env = JaxAntMaze(cfg, 1, 2)
    assert env._physics_step is mjx.step
    assert isinstance(env.reset().data, mjx.Data)
    carried = JaxAntMaze(replace(cfg, carried_state="next_step_inputs"), 1, 2).reset().data
    assert isinstance(carried, CarriedPhysics)
    shapes = [getattr(carried, name).shape for name in CARRIED_NAMES]
    assert shapes == [(2, 15), (2, 14), (2, 14), (2,)], shapes
    refused = (dict(solver_loop="scan"), dict(carried_state="nothing"),
               dict(solver_loop="fixed_count", solver_iterations=1),
               dict(solver_loop="fixed_count", integrator="RK4"))
    for changes in refused:
        try:
            JaxAntMaze(replace(cfg, **changes), 1, 1)
        except ValueError:
            continue
        raise AssertionError(f"{changes} was not refused")
    print("ok test_defaults_are_the_stepper_as_shipped")


def landed_state(env, key, steps=20):
    """The environment state after `steps` random-torque steps, by when the ants have landed."""
    step, state = jax.jit(env.step), env.reset()
    for k in range(steps):
        state, *_ = step(state, torques(key, k, env))
    return state


def largest_relative_difference(a, b):
    """max |x - y| / (1 + |y|) over every array of two MJX data trees (NaN on both sides: equal)."""
    worst = 0.0
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
        x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
        if x.size and not np.array_equal(x, y, equal_nan=True):
            worst = max(worst, float(np.nanmax(np.abs(x - y) / (1.0 + np.abs(y)))))
    return worst


def test_fixed_count_step_against_mjx_step_in_float32():
    """From 50 states with contacts, one float32 step with the fixed-count loop is mjx.step's
    to rounding."""
    env = JaxAntMaze(preset("umaze"), 8, 1)
    key = jax.random.PRNGKey(21)
    state = landed_state(env, key)

    @jax.jit
    def both(data):
        """One physics step of every environment computed both ways from the same data."""
        with jax.enable_x64(False):
            return (jax.vmap(mjx.step, in_axes=(None, 0))(env.model, data),
                    jax.vmap(physics_step_fixed_count, in_axes=(None, 0))(env.model, data))

    # GOAL: 50 physics steps along the mjx.step trajectory, every array of the data compared.
    data, steps_with_contact, worst = state.data, 0, 0.0
    for k in range(50):
        data = data.replace(ctrl=torques(key, 1000 + k, env).reshape(env.B, env.nu))
        reference, fixed = both(data)
        near = np.asarray(reference._impl.contact.dist < reference._impl.contact.includemargin)
        steps_with_contact += int(near.any())
        worst = max(worst, largest_relative_difference(fixed, reference))
        data = reference
    print(f"fixed-count step against mjx.step, float32: {steps_with_contact} of 50 states with a "
          f"contact, largest relative difference {worst:.2e}")
    assert steps_with_contact >= 25, steps_with_contact
    assert worst < 1e-4, worst
    print("ok test_fixed_count_step_against_mjx_step_in_float32")


def test_fixed_count_forms_compiled_on_their_own_in_float32():
    """Each form of the fixed-count loop, compiled as its own program as the environment does,
    against mjx.step compiled as its own program: one float32 step from the same 50 states.

    Compiled apart, the programs round differently, and in a state where the allowed passes do
    not settle which constraints are active that can change the accelerations by more than
    rounding. The gate bounds how often: at most 5% of environment-steps above 1e-4 relative.
    """
    env = JaxAntMaze(preset("umaze"), 16, 1)
    key = jax.random.PRNGKey(21)
    state = landed_state(env, key)

    def compiled(physics_step):
        """One physics step of every environment as its own compiled program."""
        def call(data):
            with jax.enable_x64(False):
                return jax.vmap(physics_step, in_axes=(None, 0))(env.model, data)
        return jax.jit(call)

    steppers = {"passes written out": compiled(physics_step_fixed_count),
                "passes in one scan": compiled(functools.partial(physics_step_fixed_count,
                                                                 written_out=False))}
    step_reference = compiled(mjx.step)
    notable = {name: 0 for name in steppers}
    unequal = {name: 0 for name in steppers}
    worst = {name: 0.0 for name in steppers}
    data = state.data
    for k in range(50):
        data = data.replace(ctrl=torques(key, 1000 + k, env).reshape(env.B, env.nu))
        reference = step_reference(data)
        for name, stepper in steppers.items():
            other = stepper(data)
            # before: the three arrays the next step reads, [B, .] each; after: per environment
            # the largest relative difference over them
            pairs = [(np.asarray(getattr(other, n)), np.asarray(getattr(reference, n)))
                     for n in ("qpos", "qvel", "qacc_warmstart")]
            gaps = np.max([np.max(np.abs(x - y) / (1.0 + np.abs(y)), axis=1) for x, y in pairs],
                          axis=0)
            notable[name] += int((gaps > 1e-4).sum())
            unequal[name] += int((gaps > 0).sum())
            worst[name] = max(worst[name], float(gaps.max()))
        data = reference
    for name in steppers:
        print(f"fixed-count loop, {name}, compiled on its own, float32, arrays the next step "
              f"reads: {unequal[name]} of {50 * env.B} environment-steps not equal, "
              f"{notable[name]} above 1e-4, largest relative difference {worst[name]:.2e}")
        assert notable[name] <= 0.05 * 50 * env.B, (name, notable[name])
    print("ok test_fixed_count_forms_compiled_on_their_own_in_float32")


def test_fixed_count_step_is_the_same_algorithm_in_float64():
    """In float64, one step with each loop from the same 50 states agrees to 1e-10 in every array,
    and 50 steps run freely agree to 1e-8 in the arrays the next step reads: the difference seen
    in float32 is rounding, not a different computation."""
    env = JaxAntMaze(preset("umaze"), 8, 1)
    key = jax.random.PRNGKey(21)
    state = landed_state(env, key)
    # GOAL: the same model and the landed states in float64 (the platform's global setting).
    mj_model = mujoco.MjModel.from_xml_string(build_antmaze_xml(env.cfg))
    model = mjx.put_model(mj_model)
    data = jax.vmap(lambda _: mjx.make_data(model))(jnp.arange(env.B))
    data = data.replace(qpos=jnp.asarray(state.data.qpos, jnp.float64),
                        qvel=jnp.asarray(state.data.qvel, jnp.float64),
                        qacc_warmstart=jnp.asarray(state.data.qacc_warmstart, jnp.float64))
    assert data.qpos.dtype == jnp.float64
    step_reference = jax.jit(jax.vmap(mjx.step, in_axes=(None, 0)))
    step_fixed = jax.jit(jax.vmap(physics_step_fixed_count, in_axes=(None, 0)))

    def next_inputs(d):
        """The arrays the next physics step reads."""
        return [d.qpos, d.qvel, d.qacc_warmstart]

    # GOAL: along the mjx.step trajectory, one step both ways from the same data (every array);
    # beside it, the fixed-count loop advancing its own trajectory (the arrays the next step
    # reads: on separate trajectories a contact can change slots, which moves other arrays).
    reference, free, steps_with_contact, worst_step, worst_free = data, data, 0, 0.0, 0.0
    for k in range(50):
        ctrl = jnp.asarray(torques(key, 1000 + k, env).reshape(env.B, env.nu), jnp.float64)
        stepped = step_reference(model, reference.replace(ctrl=ctrl))
        worst_step = max(worst_step, largest_relative_difference(
            step_fixed(model, reference.replace(ctrl=ctrl)), stepped))
        free = step_fixed(model, free.replace(ctrl=ctrl))
        worst_free = max(worst_free, largest_relative_difference(next_inputs(free),
                                                                 next_inputs(stepped)))
        near = np.asarray(stepped._impl.contact.dist < stepped._impl.contact.includemargin)
        steps_with_contact += int(near.any())
        reference = stepped
    print(f"fixed-count step against mjx.step, float64: {steps_with_contact} of 50 states with a "
          f"contact, largest relative difference in one step {worst_step:.2e}, after 50 steps "
          f"run freely {worst_free:.2e}")
    assert steps_with_contact >= 25, steps_with_contact
    assert worst_step < 1e-10, worst_step
    assert worst_free < 1e-8, worst_free
    print("ok test_fixed_count_step_is_the_same_algorithm_in_float64")


def test_fixed_count_environment_equals_the_default_before_the_first_contact():
    """Until the ant lands the solver has nothing to do, so the two loops give the same bits."""
    cfg = preset("umaze")
    key = jax.random.PRNGKey(22)
    default = rollout(JaxAntMaze(cfg, 2, 2), key, 3)
    fixed = rollout(JaxAntMaze(replace(cfg, solver_loop="fixed_count"), 2, 2), key, 3)
    for k, (a, b) in enumerate(zip(default, fixed)):
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x, y, err_msg=f"environment step {k}")
    print("ok test_fixed_count_environment_equals_the_default_before_the_first_contact")


def test_carried_state_is_bitwise_equal_for_100_steps():
    """Carrying only the next step's inputs changes no number in 100 steps, a reset included."""
    # a 60-step limit puts one truncation and its reset inside the 100 steps
    cfg = replace(preset("umaze"), max_episode_steps=60)
    key = jax.random.PRNGKey(23)
    for solver_loop in ("while_loop", "fixed_count"):
        whole = rollout(JaxAntMaze(replace(cfg, solver_loop=solver_loop), 2, 2), key, 100)
        carried = rollout(JaxAntMaze(replace(cfg, solver_loop=solver_loop,
                                             carried_state="next_step_inputs"), 2, 2), key, 100)
        for k, (a, b) in enumerate(zip(whole, carried)):
            for x, y in zip(a, b):
                np.testing.assert_array_equal(x, y, err_msg=f"{solver_loop}, environment step {k}")
        truncated = np.stack([row[3] for row in whole])
        assert truncated[59].all() and truncated.sum() == 4, truncated.sum()
    print("ok test_carried_state_is_bitwise_equal_for_100_steps")


def test_nan_guard_on_the_carried_arrays():
    """With both options on, a non-finite environment is respawned, counted and pays 0."""
    env = JaxAntMaze(replace(preset("umaze"), **BOTH), 1, 2)
    s = env.reset()
    qpos = np.asarray(s.data.qpos).copy()
    qpos[0, 3] = np.nan   # environment 0: one poisoned coordinate; environment 1 stays healthy
    s = s._replace(data=s.data.replace(qpos=jnp.asarray(qpos)))
    s, obs, rew, term, trunc, final = jax.jit(env.step)(s, jnp.zeros((1, 2, 8), jnp.float32))
    assert bool(trunc[0, 0]) and not bool(term[0, 0]) and float(rew[0, 0]) == 0.0
    assert bool(jnp.isfinite(final).all()) and bool(jnp.isfinite(obs).all())
    np.testing.assert_array_equal(np.asarray(s.data.qpos[0]), np.asarray(env.init_qpos))
    assert float(jnp.abs(s.data.qacc_warmstart[0]).max()) == 0.0 and float(s.data.time[0]) == 0.0
    assert int(s.nan_count[0, 0]) == 1 and int(s.nan_count[0, 1]) == 0
    assert bool(jnp.isfinite(s.data.qpos[1]).all()) and float(s.data.time[1]) > 0.0
    print("ok test_nan_guard_on_the_carried_arrays")


def test_both_options_train_end_to_end():
    """Two full training iterations with both options on: finite rewards, coverage counted."""
    cfg = PPOConfig(n_copies=2, n_envs=2, num_steps=8, prime_iterations=1, track_coverage=True)
    runner = Runner(cfg, bonus="rnd_next_state", env_cfg=replace(preset("umaze"), **BOTH))
    state, stats = runner.train(2, log_every_seconds=0, history_every=1)
    assert np.isfinite(stats["history"][-1]["reward_ext_sum_per_copy"]).all()
    assert all(np.isfinite(r["rint_mean_per_copy"]).all() for r in stats["history"])
    assert (np.asarray(stats["history"][-1]["coverage_per_copy"]) > 0).all()
    print("ok test_both_options_train_end_to_end")


if __name__ == "__main__":
    test_defaults_are_the_stepper_as_shipped()
    test_fixed_count_step_against_mjx_step_in_float32()
    test_fixed_count_forms_compiled_on_their_own_in_float32()
    test_fixed_count_step_is_the_same_algorithm_in_float64()
    test_fixed_count_environment_equals_the_default_before_the_first_contact()
    test_carried_state_is_bitwise_equal_for_100_steps()
    test_nan_guard_on_the_carried_arrays()
    test_both_options_train_end_to_end()
