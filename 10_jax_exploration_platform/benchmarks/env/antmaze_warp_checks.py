"""Correctness measurements of the Warp AntMaze environment against C MuJoCo and the JAX implementation.

Every function returns plain numbers and decides nothing. `tests/envs/test_antmaze_warp.py`
asserts on them on a compute node's processors (MuJoCo Warp on the processor device), and
`profile_antmaze_warp.py` records them from the graphics card.

The checks, by the letter the measurement plan gives them:
  a  one environment step from 50 states along a C-MuJoCo random-torque trajectory, against C
     MuJoCo of the same model and against the JAX implementation, contact-free and in-contact
     states separately (the states and the split are those of
     `tests/envs/test_antmaze_parity_cpu_mujoco.py`)
  b  the resting height after 1,200 passive environment steps
  c  copy isolation: changing what other environments do must not change a watched environment,
     also when the shared contact pool is nearly full; and a pool that is too small is reported
  e  reward, termination at the goal, truncation at the step limit, the non-finite respawn and
     the reset give the same values as the JAX environment from the same state
and the trainer's kind of program: a donated state carried through `lax.scan` with resets inside.
"""
import dataclasses
import hashlib
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE / "src"))
sys.path.insert(0, str(BASE / "tests" / "envs"))

import exploration_platform  # noqa: E402,F401  (64-bit statistics on, as in training)
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from exploration_platform.envs.antmaze.am_common import build_antmaze_xml, preset  # noqa: E402
from exploration_platform.envs.antmaze.mjx_antmaze import JaxAntMaze  # noqa: E402
from exploration_platform.envs.antmaze.warp_antmaze import WarpAntMaze  # noqa: E402

BASE_SEED = 0
F32 = jnp.float32
REFERENCE_RESTING_HEIGHT = 0.38248   # metres; spec.md, gate 2


def substream_key(name, *coords):
    """A JAX random key for one named random quantity, independent of every other quantity."""
    # before: name = "isolation torques", coords = (3,), BASE_SEED = 0
    # after:  PRNGKey(low 32 bits of sha256("0::isolation torques::3"))
    text = "::".join(str(part) for part in (BASE_SEED, name, *coords))
    return jax.random.PRNGKey(int(hashlib.sha256(text.encode()).hexdigest(), 16) & 0xFFFFFFFF)


def torques(name, steps, n_envs, act_dim=8):
    """Uniform torques in [-1, 1], shape [steps, n_envs, act_dim], one stream per environment."""
    key = substream_key(name)
    # before: per environment i, a [steps, act_dim] block from fold_in(key, i) -> [n_envs, steps, act_dim]
    # after:  axes swapped to [steps, n_envs, act_dim]; a larger batch only appends environments
    per_env = jax.vmap(lambda i: jax.random.uniform(
        jax.random.fold_in(key, i), (steps, act_dim), F32, -1.0, 1.0))(jnp.arange(n_envs))
    return jnp.swapaxes(per_env, 0, 1)


def with_state(env, state, qpos=None, qvel=None, step_count=None):
    """The environment state with positions, velocities or step counts replaced."""
    data = state.data
    if qpos is not None:
        data = data.replace(qpos=jnp.asarray(qpos, F32))
    if qvel is not None:
        data = data.replace(qvel=jnp.asarray(qvel, F32))
    state = state._replace(data=data)
    if step_count is not None:
        state = state._replace(step_count=jnp.asarray(step_count, jnp.int32).reshape(env.C, env.N))
    return state


def rollout_call(env):
    """Compiled function: env.step for every torque of [steps, B, 8]; returns the state and all outputs."""
    def call(state, torque_steps):
        """Scan the environment's own step over the torques."""
        def body(st, torque):
            """One environment step; its five outputs are stacked over the steps."""
            st, obs, reward, terminated, truncated, final_obs = env.step(
                st, torque.reshape(env.C, env.N, env.act_dim))
            return st, (obs, reward, terminated, truncated, final_obs)
        return jax.lax.scan(body, state, torque_steps)
    return jax.jit(call)


def spread(values):
    """Largest and median of a list of numbers, or None for an empty list."""
    return {"count": len(values), "largest": float(np.max(values)) if values else None,
            "median": float(np.median(values)) if values else None}


# ----------------------------------------------------------------------------------------------
# a: one environment step against C MuJoCo and against the JAX implementation
# ----------------------------------------------------------------------------------------------

def one_step_parity(n_states=50, warp_options=None, changes=None, with_jax=True, cfg=None):
    """One environment step from each of n_states C-MuJoCo states, in C, in Warp and in JAX.

    `cfg` is the specified model of all three (default: the U-maze preset; the stable physics of
    the AntMaze baseline is `test_antmaze_parity_cpu_mujoco.STABLE`).

    Returns, for contact-free and in-contact states separately (a state is in contact when any of
    the 5 physics steps of the C window saw a contact), the largest and the median of the largest
    absolute position difference: Warp against C, JAX against C, Warp against JAX.
    `changes` are configuration fields (for example fewer solver iterations) given to the Warp
    environment only: the C model and the JAX environment stay the specified model, so the
    numbers say how far the changed Warp stepper is from the environment's specification.
    with_jax=False leaves the JAX environment out.
    """
    from test_antmaze_parity_cpu_mujoco import CFG, c_env_step, c_trajectory
    cfg = CFG if cfg is None else cfg
    # GOAL: the reference states and what C MuJoCo makes of each of them in one environment step.
    warp_env = WarpAntMaze(dataclasses.replace(cfg, **(changes or {})), n_states, 1, **(warp_options or {}))
    qs, vs, acts, _, model = c_trajectory(warp_env, n_states, jax.random.PRNGKey(11), cfg)
    data = mujoco.MjData(model)
    c_next, in_contact = [], []
    for k in range(n_states):
        mujoco.mj_resetData(model, data)
        data.qpos[:], data.qvel[:] = qs[k], vs[k]
        mujoco.mj_forward(model, data)
        in_contact.append(c_env_step(model, data, acts[k], cfg.frame_skip) > 0)
        c_next.append(data.qpos.copy())
    c_next, in_contact = np.stack(c_next), np.asarray(in_contact)

    def stepped_positions(env):
        """All reference states as one batch of n_states environments, stepped once."""
        state = with_state(env, env.reset(), qs[:n_states], vs[:n_states])
        state, *_ = jax.jit(env.step)(state, jnp.asarray(acts, F32).reshape(n_states, 1, 8))
        return np.asarray(state.data.qpos, np.float64), state

    # GOAL: the largest absolute position difference per state, for the three pairs.
    warp_next, warp_state = stepped_positions(warp_env)
    pairs = {"warp_against_c": np.abs(warp_next - c_next).max(-1)}
    if with_jax:
        jax_next, _ = stepped_positions(JaxAntMaze(cfg, n_states, 1))
        pairs.update(jax_against_c=np.abs(jax_next - c_next).max(-1),
                     warp_against_jax=np.abs(warp_next - jax_next).max(-1))
    return {"states": n_states, "warp_changes": changes or {},
            "non_finite_resets": int(np.asarray(warp_state.nan_count).sum()),
            "contact_free": {name: spread(list(err[~in_contact])) for name, err in pairs.items()},
            "in_contact": {name: spread(list(err[in_contact])) for name, err in pairs.items()},
            "pool": warp_env.overflow_report(warp_state)}


def trajectory_divergence(steps=20, warp_options=None, cfg=None):
    """Shared random torques from the spawn in C MuJoCo, in Warp and in JAX (gate 2 of spec.md).

    `cfg` is the specified model of all three (default: the U-maze preset).

    Returns per implementation the largest position difference from the C trajectory before the
    first contact, over all steps, and at the last step. The gate: roundoff before the first
    contact (below 1e-4 m), and below one maze cell (4 m) over the 20 steps.
    """
    from test_antmaze_parity_cpu_mujoco import CFG, c_trajectory
    cfg = CFG if cfg is None else cfg
    envs = {"warp": WarpAntMaze(cfg, 1, 1, **(warp_options or {})), "jax": JaxAntMaze(cfg, 1, 1)}
    qs, _, acts, ncons, _ = c_trajectory(envs["jax"], steps, jax.random.PRNGKey(5), cfg)
    first_contact = int(np.argmax(ncons > 0)) if (ncons > 0).any() else len(ncons)
    result = {"steps": steps, "first_state_with_a_contact": first_contact}
    for name, env in envs.items():
        state, step, errors = env.reset(), jax.jit(env.step), []
        for k in range(steps):
            state, *_ = step(state, jnp.asarray(acts[k], F32).reshape(1, 1, 8))
            errors.append(float(np.abs(np.asarray(state.data.qpos[0], np.float64) - qs[k + 1]).max()))
        result[name] = {"largest_before_the_first_contact": max(errors[:max(first_contact - 1, 1)]),
                        "largest_over_all_steps": max(errors), "at_the_last_step": errors[-1],
                        "all_finite": bool(np.isfinite(errors).all())}
    return result


# ----------------------------------------------------------------------------------------------
# b: the resting height
# ----------------------------------------------------------------------------------------------

def c_settled_state(cfg, steps, init_qpos):
    """The C-MuJoCo state after `steps` passive environment steps from the spawn: (qpos, qvel)."""
    model = mujoco.MjModel.from_xml_string(build_antmaze_xml(cfg))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    data.qpos[:] = np.asarray(init_qpos, np.float64)
    for _ in range(steps * cfg.frame_skip):
        mujoco.mj_step(model, data)
    return data.qpos.copy(), data.qvel.copy()


def resting_height(steps=1200, handover_steps=200, warp_options=None, changes=None, with_jax=True):
    """Passive steps from the spawn in Warp, in JAX and in C, and a C-settled state kept by each.

    Returns per implementation the torso height and the largest speed after `steps` environment
    steps, and how far a state settled in C MuJoCo moves in `handover_steps` further passive steps.
    `changes` are configuration fields (for example fewer solver iterations) applied to all three;
    with_jax=False leaves the JAX implementation out.
    """
    # the step limit is lifted so that no reset happens while the ant settles
    cfg = dataclasses.replace(preset("umaze"), max_episode_steps=10 ** 6, **(changes or {}))
    envs = {"warp": WarpAntMaze(cfg, 1, 1, **(warp_options or {}))}
    if with_jax:
        envs["jax"] = JaxAntMaze(cfg, 1, 1)
    c_qpos, c_qvel = c_settled_state(cfg, steps, envs["warp"].init_qpos)
    result = {"steps": steps, "reference_height": REFERENCE_RESTING_HEIGHT,
              "c_mujoco": {"height": float(c_qpos[2]), "largest_speed": float(np.abs(c_qvel).max())}}
    for name, env in envs.items():
        call = rollout_call(env)
        zero = jnp.zeros((100, 1, 8), F32)
        # GOAL: settle from the spawn, 100 passive environment steps per call.
        state = env.reset()
        for _ in range(steps // 100):
            state, _ = call(state, zero)
        qpos, qvel = np.asarray(state.data.qpos[0], np.float64), np.asarray(state.data.qvel[0])
        # GOAL: start from the C-settled state and see how far this implementation moves it.
        kept = with_state(env, env.reset(), c_qpos[None], c_qvel[None])
        for _ in range(handover_steps // 100):
            kept, _ = call(kept, zero)
        if name == "warp":
            result["warp_pool"] = env.overflow_report(state)
        result[name] = {"height": float(qpos[2]), "largest_speed": float(np.abs(qvel).max()),
                        "largest_position_difference_from_c_settled_state": float(np.abs(qpos - c_qpos).max()),
                        "c_settled_state_moved_by": float(
                            np.abs(np.asarray(kept.data.qpos[0], np.float64) - c_qpos).max())}
    return result


# ----------------------------------------------------------------------------------------------
# c: copy isolation and the shared contact pool
# ----------------------------------------------------------------------------------------------

def copy_isolation(n_envs, steps=40, watch_every=8, pool_margin=None, warp_options=None, implementation="warp"):
    """Watched environments get the same torques in two runs while all others get different ones.

    Run A and run B start at the spawn. Every `watch_every`-th environment is watched and gets
    the same torques in both; every other environment gets its run-A torques times -0.5 in run B.
    A third run repeats run A. With pool_margin (for example 1.03) the shared pool of the Warp
    environment is sized that factor above the largest use a first run with the default pool saw,
    so it is nearly full. implementation="jax" runs the same three runs in the JAX environment.

    Returns, for A against B and for A against its repeat: how many watched environments have
    bitwise-equal observations at every step, the first step at which one differs, and the
    largest absolute difference among them after 1, 5, 10, 20 and all steps; and the pool use.
    """
    cfg = preset("umaze")
    options = dict(warp_options or {})
    act = torques("isolation torques", steps, n_envs)
    watched = np.arange(n_envs) % watch_every == 0
    # before: act [steps, n_envs, 8]; after: the same for watched environments, times -0.5 for the others
    act_b = jnp.where(jnp.asarray(watched)[None, :, None], act, -0.5 * act)

    def make(**more):
        """The environment of the chosen implementation with n_envs copies of one environment."""
        return WarpAntMaze(cfg, n_envs, 1, **options, **more) if implementation == "warp" else JaxAntMaze(cfg, n_envs, 1)

    def run(env, call, torque_steps):
        """The observation of every environment after every step [steps, n_envs, 29], and the final state."""
        state, outs = call(env.reset(), torque_steps)
        return np.asarray(outs[4]).reshape(steps, n_envs, 29), state

    # GOAL: with a margin, first measure the pool use, then size the pool just above it.
    result = {"implementation": implementation, "environments": n_envs, "steps": steps, "watched": int(watched.sum())}
    more = {}
    if pool_margin is not None:
        probe = make()
        probe_call = rollout_call(probe)
        report = probe.overflow_report(run(probe, probe_call, act)[1])
        report_b = probe.overflow_report(run(probe, probe_call, act_b)[1])
        peak = max(report["nacon_peak"], report["ncollision_peak"],
                   report_b["nacon_peak"], report_b["ncollision_peak"])
        more["naconmax"] = int(np.ceil(peak * pool_margin))
        result["pool_sized_from_peak"] = peak
    env = make(**more)
    call = rollout_call(env)
    out_a, state_a = run(env, call, act)
    out_b, state_b = run(env, call, act_b)
    out_a2, _ = run(env, call, act)

    def compare(x, y):
        """Watched environments of two runs: bitwise equality, first differing step, differences by step."""
        # before: x, y [steps, n_envs, 29]; after: per step the largest |x - y| over watched environments
        by_step = np.abs(x[:, watched] - y[:, watched]).max(axis=(1, 2))
        differing = np.flatnonzero((x[:, watched] != y[:, watched]).any(axis=(1, 2)))
        marks = [m for m in (1, 5, 10, 20, steps) if m <= steps]
        return {"bitwise_equal": int((x[:, watched] == y[:, watched]).all(axis=(0, 2)).sum()),
                "first_step_with_a_difference": int(differing[0]) + 1 if differing.size else None,
                "largest_difference_after_steps": {str(m): float(by_step[:m].max()) for m in marks},
                "largest_difference": float(by_step.max())}

    result.update(
        others_changed=compare(out_a, out_b), same_run_repeated=compare(out_a, out_a2),
        unwatched_environments_that_differ=int((out_a[-1, ~watched] != out_b[-1, ~watched]).any(-1).sum()),
        all_finite=bool(np.isfinite(out_a).all() and np.isfinite(out_b).all()))
    if implementation == "warp":
        result.update(pool_run_a=env.overflow_report(state_a), pool_run_b=env.overflow_report(state_b))
        result["pool_fill"] = max(result["pool_run_a"]["nacon_peak"], result["pool_run_a"]["ncollision_peak"],
                                  result["pool_run_b"]["nacon_peak"], result["pool_run_b"]["ncollision_peak"]
                                  ) / env.naconmax
    return result


def small_pool_is_reported(n_envs=8, steps=40):
    """A pool of 2 slots per environment, far below the need: the overflow report must say so."""
    env = WarpAntMaze(preset("umaze"), n_envs, 1, contacts_per_env=2)
    state, _ = rollout_call(env)(env.reset(), torques("isolation torques", steps, n_envs))
    return env.overflow_report(state)


# ----------------------------------------------------------------------------------------------
# e: reward, termination, truncation, non-finite respawn and reset against the JAX environment
# ----------------------------------------------------------------------------------------------

def semantics_against_jax(continuing_task, warp_options=None):
    """Both environments from the same five states, two zero-torque steps; lists every mismatch.

    Environment 0 starts 0.2 m from the goal, 1 starts 0.6 m from it, 2 is at the spawn one step
    before the step limit, 3 has a non-finite hinge angle, 4 is at the spawn.
    Exact fields: reward, terminated, truncated, step count, episode count, non-finite count, and
    the observation of every environment that was reset. The observation before the reset is
    compared within a tolerance, since the two physics steps differ in the last digits.
    """
    cfg = dataclasses.replace(preset("umaze"), continuing_task=continuing_task)
    envs = {"warp": WarpAntMaze(cfg, 1, 5, **(warp_options or {})), "jax": JaxAntMaze(cfg, 1, 5)}
    outputs = {}
    for name, env in envs.items():
        # GOAL: the five start states.
        state = env.reset()
        goal = np.asarray(env.goal)
        qpos = np.asarray(state.data.qpos).copy()
        qpos[0, 0:2] = goal + np.array([0.2, 0.0], np.float32)
        qpos[1, 0:2] = goal + np.array([0.6, 0.0], np.float32)
        qpos[3, 8] = np.nan
        step_count = np.zeros((1, 5), np.int32)
        step_count[0, 2] = cfg.max_episode_steps - 1
        state = with_state(env, state, qpos, step_count=step_count)
        # GOAL: two steps, keeping everything the step returns.
        step, records = jax.jit(env.step), []
        for _ in range(2):
            state, obs, reward, terminated, truncated, final_obs = step(state, jnp.zeros((1, 5, 8), F32))
            records.append({"reward": reward, "terminated": terminated, "truncated": truncated,
                            "step_count": state.step_count, "reset_count": state.reset_count,
                            "nan_count": state.nan_count, "obs": obs, "final_obs": final_obs})
        outputs[name] = [{k: np.asarray(v) for k, v in record.items()} for record in records]
    # GOAL: compare field by field.
    mismatches, largest = [], 0.0
    for k, (w, j) in enumerate(zip(outputs["warp"], outputs["jax"])):
        for field in ("reward", "terminated", "truncated", "step_count", "reset_count", "nan_count"):
            if not np.array_equal(w[field], j[field]):
                mismatches.append(f"step {k + 1} {field}: warp {w[field].tolist()} jax {j[field].tolist()}")
        was_reset = (w["terminated"] | w["truncated"])[0]
        if not np.array_equal(w["obs"][0, was_reset], j["obs"][0, was_reset]):
            mismatches.append(f"step {k + 1}: observation after a reset differs")
        for field in ("obs", "final_obs"):
            if not (np.isfinite(w[field]).all() and np.isfinite(j[field]).all()):
                mismatches.append(f"step {k + 1} {field}: a non-finite value reached the output")
            largest = max(largest, float(np.nanmax(np.abs(w[field] - j[field]))))
    return {"continuing_task": continuing_task, "mismatches": mismatches,
            "largest_observation_difference": largest,
            "warp_first_step": {k: v.tolist() for k, v in outputs["warp"][0].items() if v.ndim == 2}}


POISONS = (("qpos", 0, "torso x"), ("qpos", 3, "torso quaternion w"), ("qpos", 8, "a hinge angle"),
           ("qvel", 0, "torso x velocity"), ("qvel", 4, "torso angular velocity"), ("qvel", 8, "a hinge velocity"))


def non_finite_inputs(warp_options=None):
    """A NaN or an infinity written into one coordinate of one environment: is that environment respawned?

    Twelve poisoned environments (six coordinates, NaN and infinity) and one healthy one step once in
    each implementation. Returns per case whether each implementation counted a non-finite respawn,
    and whether the healthy environment's numbers equal those of a run in which nothing is poisoned.
    """
    cfg, n = preset("umaze"), 2 * len(POISONS) + 1
    cases = [(field, index, label, value) for field, index, label in POISONS for value in (np.nan, np.inf)]
    counted, healthy_equal = {}, {}
    for name, env in (("warp", WarpAntMaze(cfg, 1, n, **(warp_options or {}))), ("jax", JaxAntMaze(cfg, 1, n))):
        step = jax.jit(env.step)
        clean = env.reset()
        arrays = {"qpos": np.asarray(clean.data.qpos).copy(), "qvel": np.asarray(clean.data.qvel).copy()}
        # before: every environment at the spawn; after: environment k has case k's value in one coordinate
        for k, (field, index, _, value) in enumerate(cases):
            arrays[field][k, index] = value
        poisoned, *_ = step(with_state(env, clean, arrays["qpos"], arrays["qvel"]), jnp.zeros((1, n, 8), F32))
        reference, *_ = step(clean, jnp.zeros((1, n, 8), F32))
        counted[name] = np.asarray(poisoned.nan_count)[0]
        healthy_equal[name] = bool(np.array_equal(np.asarray(poisoned.data.qpos[-1]), np.asarray(reference.data.qpos[-1]))
                                   and np.array_equal(np.asarray(poisoned.data.qvel[-1]), np.asarray(reference.data.qvel[-1])))
    return {"cases": [{"value": str(value), "coordinate": label, "warp_respawned": bool(counted["warp"][k]),
                       "jax_respawned": bool(counted["jax"][k])} for k, (_, _, label, value) in enumerate(cases)],
            "healthy_environment_respawned": {name: bool(c[-1]) for name, c in counted.items()},
            "healthy_environment_unchanged": healthy_equal}


# ----------------------------------------------------------------------------------------------
# the trainer's kind of program: a donated state carried through a scan, with resets inside
# ----------------------------------------------------------------------------------------------

def captured_graphs():
    """How many CUDA graphs the Warp foreign calls of this process hold (0 on processors)."""
    from mujoco.mjx.third_party.warp._src.jax import ffi as warp_ffi
    return sum(len(callable_.captures) for callable_ in warp_ffi._FFI_CALLABLE_REGISTRY.values())


def trainer_like_rollout(n_copies, n_envs, rollout_steps, calls, episode_steps=20, warp_options=None,
                         scan_unroll=1):
    """A donated environment state carried through `calls` scans of `rollout_steps` steps.

    The step limit is `episode_steps`, so every environment is reset inside the scans. The program
    is the trainer's shape: jax.jit with the state donated, a lax.scan over the steps (unrolled
    `scan_unroll` steps at a time; the trainer's default is 16), the next state fed to the next
    call. Returns the seconds of every call, the resets counted against the number the step limit
    implies, the data types, the pool use and the graphs captured.
    """
    cfg = dataclasses.replace(preset("umaze"), max_episode_steps=episode_steps)
    env = WarpAntMaze(cfg, n_copies, n_envs, **(warp_options or {}))

    def rollout(state, obs, torque_steps):
        """The scan of the trainer's rollout: carry (state, observation), store the outputs."""
        def body(carry, torque):
            """One step; the stored fields are those the trainer keeps."""
            st, ob = carry
            stepped, next_obs, reward, terminated, truncated, final_obs = env.step(st, torque)
            return (stepped, next_obs), (ob, final_obs, reward, terminated, terminated | truncated)
        (state, obs), stored = jax.lax.scan(body, (state, obs), torque_steps, unroll=scan_unroll)
        # a 64-bit statistic beside the 32-bit physics, as the trainer keeps
        return state, obs, stored[2].astype(jnp.float64).sum(), stored[4].sum()

    program = jax.jit(rollout, donate_argnums=(0, 1))
    state, obs = env.respawn(env.reset())
    seconds, graphs, dones, reward_dtype = [], [], 0, None
    for call in range(calls):
        act = torques("trainer-like torques", rollout_steps, env.B).reshape(
            rollout_steps, n_copies, n_envs, 8) * (1.0 if call % 2 == 0 else -1.0)
        jax.block_until_ready(act)
        t0 = time.perf_counter()
        state, obs, reward_sum, done_sum = program(state, obs, act)
        jax.block_until_ready((state, obs))
        seconds.append(time.perf_counter() - t0)
        graphs.append(captured_graphs())
        dones += int(done_sum)
        reward_dtype = str(reward_sum.dtype)
    total_steps = rollout_steps * calls
    return {"copies": n_copies, "environments_per_copy": n_envs, "rollout_steps": rollout_steps,
            "calls": calls, "episode_steps": episode_steps, "scan_unroll": scan_unroll,
            "seconds_per_call": seconds,
            "graphs_captured_after_each_call": graphs,
            "resets_counted": int(np.asarray(state.reset_count).sum()), "dones_in_outputs": dones,
            "resets_expected_from_the_step_limit": env.B * (total_steps // episode_steps),
            "non_finite_resets": int(np.asarray(state.nan_count).sum()),
            "step_count_expected": total_steps % episode_steps,
            "step_count_values": sorted(set(np.asarray(state.step_count).reshape(-1).tolist())),
            "dtypes": {"qpos": str(state.data.qpos.dtype), "observation": str(obs.dtype),
                       "statistic": reward_dtype, "x64_enabled": bool(jax.config.jax_enable_x64)},
            "compiled_programs": program._cache_size(),
            "pool": env.overflow_report(state)}


def platform_training_iterations(implementation, n_copies, n_envs, num_steps=128, timed_iterations=5,
                                 episode_steps=None):
    """The platform's own training iteration with one implementation of the AntMaze physics.

    The program is the study's: the `antmaze_current` rules, the clipped update in 4 epochs of 4
    minibatches, random network distillation on the next state, the episode ending at the goal,
    the episode log. For Warp, `WarpAntMaze` is put into the composer's table of environment
    classes for the length of the composition; no file of the trainer is edited. Returns the
    seconds of every iteration (the first two include compilation), whether the loss stayed
    finite, the episode log's counts, and for Warp the pool use.
    """
    from exploration_platform.agents.ppo.config import PPOConfig
    from exploration_platform.envs.antmaze.am_common import AntMazeConfig
    from exploration_platform.training import compose
    from exploration_platform.training.runner import Runner
    limit = {} if episode_steps is None else {"max_episode_steps": episode_steps}
    env_cfg = dataclasses.replace(preset("umaze"), continuing_task=False, **limit)
    cfg = PPOConfig(configuration_preset="antmaze_current", n_copies=n_copies, n_envs=n_envs,
                    num_steps=num_steps, prime_iterations=1, update_style="epoch_minibatch", clip_coef=0.1,
                    ent_coef=0.001, learning_rate=1e-4, episode_log=True, gather_minibatches_per_epoch=True)
    # GOAL: compose the trainer with the chosen environment class, then put the table back.
    original = compose.ENV_CLASSES[AntMazeConfig]
    compose.ENV_CLASSES[AntMazeConfig] = WarpAntMaze if implementation == "warp" else original
    try:
        runner = Runner(cfg, bonus="rnd_next_state", env_cfg=env_cfg)
    finally:
        compose.ENV_CLASSES[AntMazeConfig] = original
    # GOAL: prime, then iterate; every iteration is timed to its end.
    state = runner.prime(runner.init_state(run_seed=0))
    total, seconds, losses = 2 + timed_iterations, [], []
    for k in range(1, total + 1):
        t0 = time.perf_counter()
        state, metrics = runner.iterate(state, runner.lr_argument(k, total))
        jax.block_until_ready(metrics["loss"])
        seconds.append(time.perf_counter() - t0)
        losses.append(np.asarray(metrics["loss"]))
    log = {name: np.asarray(value) for name, value in state.agent_state["episode_log"].items()}
    env_state = state.env_state
    result = {"implementation": implementation, "environment_class": type(runner.env).__name__,
              "copies": n_copies, "environments_per_copy": n_envs, "rollout_steps": num_steps,
              "scan_unroll": cfg.scan_unroll, "iterations": total, "seconds_per_iteration": seconds,
              "loss_finite": bool(all(np.isfinite(loss).all() for loss in losses)),
              "episodes_ended": int(log["episodes"].sum()), "arrivals": int(log["arrivals"].sum()),
              "non_finite_resets": int(np.asarray(env_state.nan_count).sum()),
              "physics_dtype": str(env_state.data.qpos.dtype),
              "graphs_captured": captured_graphs() if implementation == "warp" else None}
    if implementation == "warp":
        result["pool"] = runner.env.overflow_report(env_state)
    return result
