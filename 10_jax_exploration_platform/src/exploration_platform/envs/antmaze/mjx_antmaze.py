"""Batched AntMaze in JAX, through MuJoCo MJX — the platform's second environment family.

The ant is real articulated physics, so unlike PointMaze the stepper is not hand-derived: the
model is the vendored Gymnasium `ant.xml` plus the maze walls (`am_common.build_antmaze_xml`),
and one environment step is `frame_skip` MJX physics steps, vmapped over every environment of
every copy. Resets are fully deterministic (zero reset noise on both the cells and the joints,
matching AntMaze-v5's `reset_noise_scale=0.0` plus this platform's zero position noise), so
auto-reset is a `where` against the initial state and the environment holds no RNG at all.

The platform runs with `jax_enable_x64=True` globally (float64 running statistics); MJX under
that default would build a float64 physics program, twice the memory and well below half the
speed. Every MJX call here is therefore wrapped in `jax.enable_x64(False)`, which is read at
trace time, so the compiled iteration contains a float32 physics program inside the platform's
otherwise-unchanged float64-statistics world. `tests/envs/test_antmaze_mjx.py` checks that the
boundary holds.

Two configuration fields change how the same physics is run, not what it computes
(`AntMazeConfig.solver_loop` and `AntMazeConfig.carried_state`; both default to the stepper as it
was before they existed). `tests/envs/test_antmaze_mjx_options.py` compares each with the default.
"""
import functools
from typing import NamedTuple

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx
from mujoco.mjx._src import math as mjx_math
from mujoco.mjx._src import solver as mjx_solver

from .am_common import SCALING, AntMazeConfig, build_antmaze_xml, cell_center
from ..pointmaze.pm_common import MAPS

F32 = jnp.float32
SOLVER_LOOPS = ("while_loop", "fixed_count", "fixed_count_scan")
CARRIED_STATES = ("whole_data", "next_step_inputs")


class CarriedPhysics(NamedTuple):
    """The physics arrays one step hands to the next under carried_state="next_step_inputs"."""
    qpos: jnp.ndarray             # [B, 15] positions
    qvel: jnp.ndarray             # [B, 14] velocities
    qacc_warmstart: jnp.ndarray   # [B, 14] the solver's start values: the last step's accelerations
    time: jnp.ndarray             # [B] the physics clock, seconds

    def replace(self, **changes):
        """A copy with some arrays replaced, the call mjx.Data offers under the same name."""
        return self._replace(**changes)


class EnvState(NamedTuple):
    """All mutable env state. data is the batched physics state over B = C*N environments."""
    data: object               # mjx.Data or CarriedPhysics, every array leading axis [C*N]
    step_count: jnp.ndarray    # [C, N] int32, steps since this environment's last reset
    reset_count: jnp.ndarray   # [C, N] int32, completed episodes (deterministic resets: metric only)
    nan_count: jnp.ndarray     # [C, N] int32, episodes ended by a non-finite physics state


def wants_another_pass(model, ctx):
    """mjx.solve's stop test for one environment: True while neither the cost improvement of the
    last pass nor the gradient is under the solver tolerance."""
    improvement = mjx_solver._rescale(model, ctx.prev_cost - ctx.cost)
    gradient = mjx_solver._rescale(model, mjx_math.norm(ctx.grad))
    return ~((improvement < model.opt.tolerance) | (gradient < model.opt.tolerance))


def newton_passes(model, data, written_out=True):
    """The solver context of one environment after mjx.solve's Newton passes, run in a loop of
    fixed length.

    mjx.solve runs the passes in jax.lax.while_loop, which under jax.vmap repeats while any
    environment still wants a pass and gives every environment a select between its new and its
    old values. This function is the same arithmetic in a loop of opt.iterations passes: every
    pass is computed, and an environment whose stop test is met keeps its values. With
    written_out the passes are separate copies in the compiled program; without, they are one
    jax.lax.scan. The functions with a leading underscore are MJX's own.
    """
    m, d = model, data
    # GOAL: the start point mjx.solve chooses: the warm start when its cost is the lower one.
    warm = mjx_solver.Context.create(m, d.replace(qacc=d.qacc_warmstart), grad=False)
    smooth = mjx_solver.Context.create(m, d.replace(qacc=d.qacc_smooth), grad=False)
    d = d.replace(qacc=jnp.where(warm.cost < smooth.cost, d.qacc_warmstart, d.qacc_smooth))
    ctx = mjx_solver.Context.create(m, d)
    # GOAL: the passes of mjx.solve's loop body, each kept only where the stop test asks for it
    # (mjx.solve's other stop test, on the pass count, can never be met inside this loop).
    def one_pass(ctx, _=None):
        """One pass of mjx.solve's loop body, kept only where the stop test asks for it."""
        keep_going = wants_another_pass(m, ctx)
        new = mjx_solver._linesearch(m, d, ctx)
        new = mjx_solver._update_constraint(m, d, new)
        new = mjx_solver._update_gradient(m, d, new)
        new = new.replace(search=-new.Mgrad, solver_niter=new.solver_niter + 1)
        return jax.tree.map(lambda n, o: jnp.where(keep_going, n, o), new, ctx), None

    if written_out:
        for _ in range(m.opt.iterations):
            ctx, _ = one_pass(ctx)
        return ctx
    return jax.lax.scan(one_pass, ctx, None, length=m.opt.iterations)[0]


def solve_fixed_count(model, data, written_out=True):
    """mjx.solve for one environment through newton_passes.

    Only the constraint force, the acceleration and the constraint-space force leave the solver,
    so with the passes written out the gradient and the search direction of the last pass are
    never read and the compiler drops their computation.
    """
    ctx = newton_passes(model, data, written_out)
    return data.tree_replace({"qfrc_constraint": ctx.qfrc_constraint, "qacc": ctx.qacc,
                              "_impl.efc_force": ctx.efc_force})


def physics_step_fixed_count(model, data, written_out=True):
    """mjx.step for one environment (implicitfast), from MJX's stages and solve_fixed_count."""
    d = mjx.fwd_position(model, data)
    d = mjx.sensor_pos(model, d)
    d = mjx.fwd_velocity(model, d)
    d = mjx.sensor_vel(model, d)
    d = mjx.fwd_actuation(model, d)
    d = mjx.fwd_acceleration(model, d)
    d = solve_fixed_count(model, d, written_out)
    d = mjx.sensor_acc(model, d)
    return mjx.implicit(model, d)


class JaxAntMaze:
    """Holds the compiled-in model and configuration and exposes pure reset/step functions."""

    # the platform composer reads these to size the agent, the bonus and the noise draws
    obs_dim = 29
    act_dim = 8

    def __init__(self, cfg: AntMazeConfig, n_copies: int, n_envs: int, base_seed: int = 0,
                 dtype=jnp.float32, copy_seed_index=None):
        """Build the MJX model and the deterministic initial state for one (C, N) shape.

        base_seed / copy_seed_index are accepted for the composer's uniform call signature and
        unused: with zero reset noise this environment draws no random numbers.
        """
        if dtype is not jnp.float32:
            raise ValueError("JaxAntMaze runs MJX in float32; float64 physics is not supported")
        if cfg.solver_loop not in SOLVER_LOOPS:
            raise ValueError(f"unknown solver_loop {cfg.solver_loop!r}; choose from {SOLVER_LOOPS}")
        if cfg.carried_state not in CARRIED_STATES:
            raise ValueError(f"unknown carried_state {cfg.carried_state!r}; choose from "
                             f"{CARRIED_STATES}")
        if cfg.solver_loop != "while_loop" and (cfg.integrator != "implicitfast"
                                                or cfg.solver_iterations < 2):
            raise ValueError("a fixed-count solver loop reproduces mjx.step only for the "
                             "implicitfast integrator with 2 or more solver iterations (with 1 "
                             f"MJX runs no loop); got integrator {cfg.integrator!r}, "
                             f"solver_iterations {cfg.solver_iterations}")
        self.cfg, self.C, self.N, self.dtype = cfg, n_copies, n_envs, dtype
        self.B = n_copies * n_envs

        maze_map = MAPS[cfg.map_name]
        self.rows, self.cols = len(maze_map), len(maze_map[0])
        self.n_cells = self.rows * self.cols
        self.open_cells = jnp.asarray(np.asarray(maze_map).reshape(-1) == 0)

        mj_model = mujoco.MjModel.from_xml_string(build_antmaze_xml(cfg))
        with jax.enable_x64(False):
            self.model = mjx.put_model(mj_model)
            if cfg.carried_state == "next_step_inputs":
                # one environment's MJX data: the constant every physics step starts from
                self._blank_data = mjx.make_data(self.model)
        self.nq, self.nv, self.nu = mj_model.nq, mj_model.nv, mj_model.nu
        self._physics_step = {
            "while_loop": mjx.step, "fixed_count": physics_step_fixed_count,
            "fixed_count_scan": functools.partial(physics_step_fixed_count, written_out=False),
        }[cfg.solver_loop]

        # the deterministic spawn: the model's qpos0 (torso z = 0.75, identity orientation,
        # zero joint angles) with the torso xy moved to the start cell's centre; zero velocity
        start_xy = cell_center(cfg.start_cell, self.rows, self.cols)
        qpos0 = np.asarray(mj_model.qpos0, dtype=np.float32)
        qpos0[0:2] = start_xy
        self.init_qpos = jnp.asarray(qpos0)
        self.init_qvel = jnp.zeros((self.nv,), F32)
        self.goal = jnp.asarray(cell_center(cfg.goal_cell, self.rows, self.cols), F32)

    def _spawned_data(self):
        """Fresh batched physics state at the deterministic spawn, [B] on every leading axis."""
        with jax.enable_x64(False):
            if self.cfg.carried_state == "next_step_inputs":
                # four separate arrays: donation refuses a buffer that appears twice in the tree
                return CarriedPhysics(jnp.tile(self.init_qpos[None], (self.B, 1)),
                                      jnp.tile(self.init_qvel[None], (self.B, 1)),
                                      jnp.zeros((self.B, self.nv), F32), jnp.zeros((self.B,), F32))
            data = jax.vmap(lambda _: mjx.make_data(self.model))(jnp.arange(self.B))
            return data.replace(qpos=jnp.tile(self.init_qpos[None], (self.B, 1)),
                                qvel=jnp.tile(self.init_qvel[None], (self.B, 1)))

    def _carried_physics_step(self, carried, ctrl):
        """One physics step of one environment from the carried arrays alone.

        before: carried = (qpos [15], qvel [14], qacc_warmstart [14], time []), ctrl [8]
        after:  the same four arrays one physics step later; every other array of the MJX data
                is computed from them inside the step and dropped
        """
        data = self._blank_data.replace(qpos=carried.qpos, qvel=carried.qvel,
                                        qacc_warmstart=carried.qacc_warmstart,
                                        time=carried.time, ctrl=ctrl)
        data = self._physics_step(self.model, data)
        return CarriedPhysics(data.qpos, data.qvel, data.qacc_warmstart, data.time)

    def reset(self) -> EnvState:
        """Initial state: every environment at the spawn, step and episode counters at zero.

        The two counters are separate arrays on purpose: the training state is donated to the
        compiled iteration, and donation refuses a buffer that appears twice in the tree.
        """
        return EnvState(self._spawned_data(), jnp.zeros((self.C, self.N), jnp.int32),
                        jnp.zeros((self.C, self.N), jnp.int32),
                        jnp.zeros((self.C, self.N), jnp.int32))

    def _obs(self, data):
        """Observation [C, N, 29]: the full qpos (15, torso xy included) then qvel (14).

        The reference splits the same numbers into a 27-d `observation` plus a 2-d
        `achieved_goal`; the platform keeps them concatenated because the exploration bonus
        scores the next state and the torso position is the part of it that says where the ant
        is (spec.md, deviations).
        """
        return jnp.concatenate([data.qpos, data.qvel], -1).reshape(self.C, self.N, 29)

    def respawn(self, state: EnvState):
        """Every environment back at the spawn with step counts zeroed; episode counter kept.

        The runner calls this once after the warm-up rollouts so training starts on full-length
        episodes, mirroring the PointMaze respawn.
        """
        state = EnvState(self._spawned_data(), jnp.zeros_like(state.step_count),
                         state.reset_count, state.nan_count)
        return state, self._obs(state.data)

    def step(self, state: EnvState, act):
        """One env step (frame_skip physics steps) with auto-reset. Returns (state', obs,
        reward, terminated, truncated, final_obs) — the platform's environment contract."""
        with jax.enable_x64(False):
            # torque clip and the frame-skip physics rollout, all float32. The frame skip is a
            # scan, not a Python loop: a loop would inline frame_skip copies of the physics
            # step into every compiled program and multiply the compile time by that factor
            # (measured: the whole benchmark grid stalled in compilation before this change)
            a = jnp.clip(act.astype(F32), -1.0, 1.0).reshape(self.B, self.nu)
            if self.cfg.carried_state == "whole_data":
                data = state.data.replace(ctrl=a)

                def frame(d, _):
                    """One physics step at the held ctrl."""
                    return jax.vmap(self._physics_step, in_axes=(None, 0))(self.model, d), None
            else:
                data = state.data

                def frame(carried, _):
                    """One physics step at the held ctrl, from the carried arrays alone."""
                    return jax.vmap(self._carried_physics_step)(carried, a), None
            data, _ = jax.lax.scan(frame, data, None, length=self.cfg.frame_skip)

            step_count = state.step_count + 1

            # a float32 contact solve can fail on a knife-edge state and leave NaNs — rare
            # (order one event in a hundred thousand copy-iterations, and not reproducible:
            # the card's non-deterministic reductions decide whether the edge tips), but one
            # poisoned copy would otherwise train on garbage forever. Such an environment is
            # respawned like a truncation, pays no reward, and is counted.
            bad = (~jnp.isfinite(data.qpos).all(-1) | ~jnp.isfinite(data.qvel).all(-1)
                   ).reshape(self.C, self.N)

            # sparse goal reward on the torso's xy after the step
            delta = data.qpos[:, 0:2].reshape(self.C, self.N, 2) - self.goal
            at_goal = (delta * delta).sum(-1) <= self.cfg.goal_radius ** 2
            reward = jnp.where(bad, 0.0, at_goal.astype(F32) + self.cfg.reward_shift)
            if self.cfg.continuing_task:
                terminated = jnp.zeros_like(at_goal)
            else:
                terminated = at_goal & ~bad
            truncated = ((step_count >= self.cfg.max_episode_steps) | bad) & ~terminated
            done = terminated | truncated

            # a broken environment's observation is sanitised to its spawn, so the batch the
            # networks see never carries a NaN
            spawn_obs = jnp.concatenate([self.init_qpos, self.init_qvel])
            final_obs = jnp.where(bad[..., None], spawn_obs, self._obs(data))

            # deterministic auto-reset: put finished environments back at the spawn. Only the
            # state the next physics step reads is reset — positions, velocities, the solver
            # warm start and the clock; everything else in `data` is recomputed by mjx.step
            # (and is not carried at all under carried_state="next_step_inputs").
            done_b = done.reshape(self.B)[:, None]
            data = data.replace(
                qpos=jnp.where(done_b, self.init_qpos[None], data.qpos),
                qvel=jnp.where(done_b, self.init_qvel[None], data.qvel),
                qacc_warmstart=jnp.where(done_b, 0.0, data.qacc_warmstart),
                time=jnp.where(done_b[:, 0], 0.0, data.time))
            step_count = jnp.where(done, 0, step_count)
            reset_count = state.reset_count + done.astype(jnp.int32)
            nan_count = state.nan_count + bad.astype(jnp.int32)

            obs = self._obs(data)
            return (EnvState(data, step_count, reset_count, nan_count),
                    obs, reward, terminated, truncated, final_obs)

    def cell_index(self, obs_flat):
        """Flat maze-cell index of [C, M, obs] observations -> [C, M] int32, for coverage.

        before: obs_flat[..., 0:2] world metres, e.g. large-map spawn (-18.0, -12.0)
        after:  row 7 * 12 cols + col 1 = 85
        """
        j = jnp.clip((obs_flat[..., 0] / SCALING + self.cols / 2.0).astype(jnp.int32),
                     0, self.cols - 1)
        i = jnp.clip((self.rows / 2.0 - obs_flat[..., 1] / SCALING).astype(jnp.int32),
                     0, self.rows - 1)
        return i * self.cols + j
