"""Batched AntMaze through MuJoCo Warp, called from JAX — a candidate replacement for `mjx_antmaze`.

The model, the frame skip, the reward, the goal termination, the truncation, the reset to the
fixed spawn, the 29-number observation and the non-finite respawn are those of
`mjx_antmaze.JaxAntMaze`; only the physics step differs. `mujoco.mjx.step` with a model put with
`impl="warp"` runs MuJoCo Warp's kernels on the card through one foreign call per physics step,
inside `jax.jit` and `jax.lax.scan`, for all C*N environments as one batch.

Three things differ from the JAX implementation and are visible here:

1. Contacts of all environments share ONE pool of `naconmax` slots (candidate geom pairs share a
   pool of the same size). When the pool is full, further contacts are dropped without an error.
   The state therefore carries the largest pool use seen so far and MuJoCo Warp's own overflow
   flags, and `overflow_report` turns them into a verdict a caller must check.
2. Constraint rows are limited per environment by `njmax`; rows beyond it are dropped and flagged
   the same way.
3. The MJX-JAX contact caps of the model (`max_geom_pairs`, `max_contact_points`) are custom
   numerics that only MJX's JAX implementation reads; MuJoCo Warp ignores them. The
   configuration fields `solver_loop` and `carried_state` are read by `mjx_antmaze` only.

Two measured facts limit its use (2026-11-12, `benchmarks/env/profile_antmaze_warp.py`):

- Only graph mode `WARP` returns the pool counters. In `WARP_STAGED` the physics is the same
  but the contact, candidate-pair and row counts read 0 at every step, so `overflow_report`
  cannot see a drop there.
- MuJoCo Warp normalises the torso quaternion at the start of a step and turns a quaternion
  that holds a NaN into a valid one. A NaN written only into the quaternion is therefore not
  respawned, where the JAX implementation respawns it; a non-finite value in any other
  coordinate is respawned by both.

The physics is float32. As in `mjx_antmaze`, every MJX call is traced inside
`jax.enable_x64(False)`, so the platform's 64-bit running statistics stay outside it.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx
from mujoco.mjx import warp as mjx_warp
from mujoco.mjx.warp import types as mjx_warp_types

from .am_common import SCALING, AntMazeConfig, build_antmaze_xml, cell_center
from ..pointmaze.pm_common import MAPS

F32 = jnp.float32
# MuJoCo Warp's overflow bits (third_party/mujoco_warp/_src/types.py, OverflowType)
OVERFLOW_BITS = {1: "constraint rows over njmax", 2: "constraint Jacobian non-zeros",
                 4: "candidate pairs over the shared pool", 8: "contacts over the shared pool",
                 16: "convex collision", 32: "height field", 64: "contact sensor matches",
                 128: "active degrees of freedom"}


class WarpEnvState(NamedTuple):
    """All mutable env state. data is the batched MJX state (Warp implementation) over B = C*N."""
    data: object                  # mjx.Data; per-environment arrays lead with [C*N], pools do not
    step_count: jnp.ndarray       # [C, N] int32, steps since this environment's last reset
    reset_count: jnp.ndarray      # [C, N] int32, completed episodes
    nan_count: jnp.ndarray        # [C, N] int32, episodes ended by a non-finite physics state
    nacon_peak: jnp.ndarray       # [] int32, most contacts in the shared pool at any physics step
    ncollision_peak: jnp.ndarray  # [] int32, most candidate pairs in the shared pool
    nefc_peak: jnp.ndarray        # [] int32, most constraint rows of one environment
    overflow: jnp.ndarray         # [C, N] int32, MuJoCo Warp's overflow bits, kept once set


class WarpAntMaze:
    """Holds the compiled-in model and configuration and exposes pure reset/step functions."""

    # the platform composer reads these to size the agent, the bonus and the noise draws
    obs_dim = 29
    act_dim = 8

    def __init__(self, cfg: AntMazeConfig, n_copies: int, n_envs: int, base_seed: int = 0,
                 dtype=jnp.float32, copy_seed_index=None, contacts_per_env: int = 48,
                 njmax: int = 160, graph_mode: str = "WARP", naconmax: int = None):
        """Build the Warp model and the deterministic spawn for one (C, N) shape.

        contacts_per_env sizes the shared pool (naconmax = C*N*contacts_per_env, unless naconmax
        is given as a whole number of slots); njmax is the constraint-row limit of one
        environment; graph_mode names how the foreign call replays its CUDA graph (`WARP`: one
        captured graph per set of buffer addresses; `WARP_STAGED`: one graph, with copies to
        fixed buffers). base_seed / copy_seed_index are accepted for the composer's uniform call
        signature and unused: this environment draws no random numbers.
        """
        if dtype is not jnp.float32:
            raise ValueError("WarpAntMaze runs MuJoCo Warp in float32; float64 physics is not supported")
        if not mjx_warp.WARP_INSTALLED:
            raise RuntimeError("WarpAntMaze needs warp-lang (MJX 3.11.0 asks for 1.14.0); this interpreter has "
                               "none. The environment /p/rlprojects/RND/.venvs/platform_jax_warp holds it.")
        self.cfg, self.C, self.N, self.dtype = cfg, n_copies, n_envs, dtype
        self.B = n_copies * n_envs
        self.naconmax = self.B * contacts_per_env if naconmax is None else naconmax
        self.njmax = njmax

        maze_map = MAPS[cfg.map_name]
        self.rows, self.cols = len(maze_map), len(maze_map[0])
        self.n_cells = self.rows * self.cols
        self.open_cells = jnp.asarray(np.asarray(maze_map).reshape(-1) == 0)

        # the same XML as the JAX implementation; make_data of the Warp implementation needs the
        # C model, so it is kept
        self.mj_model = mujoco.MjModel.from_xml_string(build_antmaze_xml(cfg))
        with jax.enable_x64(False):
            self.model = mjx.put_model(self.mj_model, impl="warp",
                                       graph_mode=getattr(mjx_warp_types.GraphMode, graph_mode))
        self.nq, self.nv, self.nu = self.mj_model.nq, self.mj_model.nv, self.mj_model.nu

        # the deterministic spawn: the model's qpos0 with the torso xy moved to the start cell's
        # centre; zero velocity
        start_xy = cell_center(cfg.start_cell, self.rows, self.cols)
        qpos0 = np.asarray(self.mj_model.qpos0, dtype=np.float32)
        qpos0[0:2] = start_xy
        self.init_qpos = jnp.asarray(qpos0)
        self.init_qvel = jnp.zeros((self.nv,), F32)
        self.goal = jnp.asarray(cell_center(cfg.goal_cell, self.rows, self.cols), F32)

    def _spawned_data(self):
        """Fresh batched Warp data at the deterministic spawn.

        Per-environment arrays get the leading axis [B]; the contact pool and the other shared
        arrays keep their own shape (MJX registers the Warp data so that vmap leaves them alone).
        """
        with jax.enable_x64(False):
            make = lambda _: mjx.make_data(self.mj_model, impl="warp", naconmax=self.naconmax,
                                           njmax=self.njmax)
            data = jax.vmap(make)(jnp.arange(self.B))
            return data.replace(qpos=jnp.tile(self.init_qpos[None], (self.B, 1)),
                                qvel=jnp.tile(self.init_qvel[None], (self.B, 1)))

    def reset(self) -> WarpEnvState:
        """Initial state: every environment at the spawn, all counters and peaks at zero.

        Every counter is its own array: the training state is donated to the compiled iteration,
        and donation refuses a buffer that appears twice in the tree.
        """
        zeros = lambda: jnp.zeros((self.C, self.N), jnp.int32)
        scalar = lambda: jnp.zeros((), jnp.int32)
        return WarpEnvState(self._spawned_data(), zeros(), zeros(), zeros(),
                            scalar(), scalar(), scalar(), zeros())

    def _obs(self, data):
        """Observation [C, N, 29]: the full qpos (15, torso xy included) then qvel (14)."""
        return jnp.concatenate([data.qpos, data.qvel], -1).reshape(self.C, self.N, 29)

    def respawn(self, state: WarpEnvState):
        """Every environment back at the spawn with step counts zeroed; other counters kept."""
        state = state._replace(data=self._spawned_data(),
                               step_count=jnp.zeros_like(state.step_count))
        return state, self._obs(state.data)

    def step(self, state: WarpEnvState, act):
        """One env step (frame_skip physics steps) with auto-reset. Returns (state', obs,
        reward, terminated, truncated, final_obs) — the platform's environment contract."""
        with jax.enable_x64(False):
            # torque clip and the frame-skip physics rollout, all float32
            a = jnp.clip(act.astype(F32), -1.0, 1.0).reshape(self.B, self.nu)
            data = state.data.replace(ctrl=a)

            def frame(d, _):
                """One physics step at the held ctrl; outputs the pool use and the row count."""
                d = jax.vmap(mjx.step, in_axes=(None, 0))(self.model, d)
                impl = d._impl
                return d, (impl.nacon.max(), impl.ncollision.max(), impl.nefc.max())
            data, (nacon, ncollision, nefc) = jax.lax.scan(frame, data, None,
                                                           length=self.cfg.frame_skip)

            step_count = state.step_count + 1

            # a non-finite state is respawned like a truncation, pays no reward, and is counted
            # (same rule as the JAX implementation)
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

            # a broken environment's observation is sanitised to its spawn
            spawn_obs = jnp.concatenate([self.init_qpos, self.init_qvel])
            final_obs = jnp.where(bad[..., None], spawn_obs, self._obs(data))

            # deterministic auto-reset of what the next physics step reads: positions,
            # velocities, the solver's start values and the clock
            done_b = done.reshape(self.B)[:, None]
            data = data.replace(
                qpos=jnp.where(done_b, self.init_qpos[None], data.qpos),
                qvel=jnp.where(done_b, self.init_qvel[None], data.qvel),
                qacc_warmstart=jnp.where(done_b, 0.0, data.qacc_warmstart),
                time=jnp.where(done_b[:, 0], 0.0, data.time))

            obs = self._obs(data)
            stepped = WarpEnvState(
                data, jnp.where(done, 0, step_count),
                state.reset_count + done.astype(jnp.int32),
                state.nan_count + bad.astype(jnp.int32),
                jnp.maximum(state.nacon_peak, nacon.max()),
                jnp.maximum(state.ncollision_peak, ncollision.max()),
                jnp.maximum(state.nefc_peak, nefc.max()),
                state.overflow | data._impl.overflow.reshape(self.C, self.N))
            return stepped, obs, reward, terminated, truncated, final_obs

    def overflow_report(self, state: WarpEnvState):
        """Pool use and limits as plain numbers, and whether any contact or row was dropped.

        before: state with nacon_peak 9,731, naconmax 49,152, nefc_peak 52, njmax 160, no flag set
        after:  {"nacon_peak": 9731, "naconmax": 49152, ..., "dropped": False, "reasons": []}
        """
        nacon, ncollision, nefc = (int(state.nacon_peak), int(state.ncollision_peak),
                                   int(state.nefc_peak))
        bits = int(np.bitwise_or.reduce(np.asarray(state.overflow).reshape(-1)))
        reasons = [name for bit, name in OVERFLOW_BITS.items() if bits & bit]
        if nacon > self.naconmax and OVERFLOW_BITS[8] not in reasons:
            reasons.append(OVERFLOW_BITS[8])
        if ncollision > self.naconmax and OVERFLOW_BITS[4] not in reasons:
            reasons.append(OVERFLOW_BITS[4])
        if nefc > self.njmax and OVERFLOW_BITS[1] not in reasons:
            reasons.append(OVERFLOW_BITS[1])
        return {"nacon_peak": nacon, "ncollision_peak": ncollision, "naconmax": self.naconmax,
                "nefc_peak": nefc, "njmax": self.njmax, "overflow_bits": bits,
                "dropped": bool(reasons), "reasons": reasons}

    def raise_on_overflow(self, state: WarpEnvState):
        """Raise if MuJoCo Warp dropped a contact, a candidate pair or a constraint row."""
        report = self.overflow_report(state)
        if report["dropped"]:
            raise RuntimeError(f"MuJoCo Warp dropped contacts or constraint rows: {report}")
        return report

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
