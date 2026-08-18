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
otherwise-unchanged float64-statistics world. `tests/envs/test_antmaze_x64_boundary.py` checks
that the boundary holds.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx

from .am_common import SCALING, AntMazeConfig, build_antmaze_xml, cell_center
from ..pointmaze.pm_common import MAPS

F32 = jnp.float32


class EnvState(NamedTuple):
    """All mutable env state. data is the batched MJX state over B = C*N environments."""
    data: object               # mjx.Data, every array leading axis [C*N]
    step_count: jnp.ndarray    # [C, N] int32, steps since this environment's last reset
    reset_count: jnp.ndarray   # [C, N] int32, completed episodes (deterministic resets: metric only)


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
        self.cfg, self.C, self.N, self.dtype = cfg, n_copies, n_envs, dtype
        self.B = n_copies * n_envs

        maze_map = MAPS[cfg.map_name]
        self.rows, self.cols = len(maze_map), len(maze_map[0])
        self.n_cells = self.rows * self.cols
        self.open_cells = jnp.asarray(np.asarray(maze_map).reshape(-1) == 0)

        mj_model = mujoco.MjModel.from_xml_string(build_antmaze_xml(cfg))
        with jax.enable_x64(False):
            self.model = mjx.put_model(mj_model)
        self.nq, self.nv, self.nu = mj_model.nq, mj_model.nv, mj_model.nu

        # the deterministic spawn: the model's qpos0 (torso z = 0.75, identity orientation,
        # zero joint angles) with the torso xy moved to the start cell's centre; zero velocity
        start_xy = cell_center(cfg.start_cell, self.rows, self.cols)
        qpos0 = np.asarray(mj_model.qpos0, dtype=np.float32)
        qpos0[0:2] = start_xy
        self.init_qpos = jnp.asarray(qpos0)
        self.init_qvel = jnp.zeros((self.nv,), F32)
        self.goal = jnp.asarray(cell_center(cfg.goal_cell, self.rows, self.cols), F32)

    def _spawned_data(self):
        """Fresh batched MJX data at the deterministic spawn, [B] on every leading axis."""
        with jax.enable_x64(False):
            data = jax.vmap(lambda _: mjx.make_data(self.model))(jnp.arange(self.B))
            return data.replace(qpos=jnp.tile(self.init_qpos[None], (self.B, 1)),
                                qvel=jnp.tile(self.init_qvel[None], (self.B, 1)))

    def reset(self) -> EnvState:
        """Initial state: every environment at the spawn, step and episode counters at zero.

        The two counters are separate arrays on purpose: the training state is donated to the
        compiled iteration, and donation refuses a buffer that appears twice in the tree.
        """
        return EnvState(self._spawned_data(), jnp.zeros((self.C, self.N), jnp.int32),
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
                         state.reset_count)
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
            data = state.data.replace(ctrl=a)

            def frame(d, _):
                """One physics step at the held ctrl."""
                return jax.vmap(mjx.step, in_axes=(None, 0))(self.model, d), None
            data, _ = jax.lax.scan(frame, data, None, length=self.cfg.frame_skip)

            step_count = state.step_count + 1

            # sparse goal reward on the torso's xy after the step
            delta = data.qpos[:, 0:2].reshape(self.C, self.N, 2) - self.goal
            at_goal = (delta * delta).sum(-1) <= self.cfg.goal_radius ** 2
            reward = at_goal.astype(F32) + self.cfg.reward_shift
            if self.cfg.continuing_task:
                terminated = jnp.zeros_like(at_goal)
            else:
                terminated = at_goal
            truncated = (step_count >= self.cfg.max_episode_steps) & ~terminated
            done = terminated | truncated

            final_obs = self._obs(data)

            # deterministic auto-reset: put finished environments back at the spawn. Only the
            # state the next physics step reads is reset — positions, velocities, the solver
            # warm start and the clock; everything else in `data` is recomputed by mjx.step.
            done_b = done.reshape(self.B)[:, None]
            data = data.replace(
                qpos=jnp.where(done_b, self.init_qpos[None], data.qpos),
                qvel=jnp.where(done_b, self.init_qvel[None], data.qvel),
                qacc_warmstart=jnp.where(done_b, 0.0, data.qacc_warmstart),
                time=jnp.where(done_b[:, 0], 0.0, data.time))
            step_count = jnp.where(done, 0, step_count)
            reset_count = state.reset_count + done.astype(jnp.int32)

            obs = self._obs(data)
            return (EnvState(data, step_count, reset_count),
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
