"""Batched PointMaze in JAX — v0. Pure-functional mirror of the torch implementation:
same probe-verified exact physics (unrolled 8-candidate contacts, exact 2-contact QP),
same FROZEN reset RNG (two fmix32 rounds; bit-identical resets across torch/CUDA/JAX).

State is a NamedTuple of arrays [C, N, ...]; step is a pure function meant to be jit-ed
with donated state. `make_step_batch` exposes the raw dynamics for the fixture checker.
"""
from typing import NamedTuple

import numpy as np

from .pm_common import (ACT_CLIP, B_REF, H, INV_MHD, K_OVER_D, M, MARGIN, NEIGHBOR_OFFSETS, R,
                        SOL_D0, SOL_DMAX, SOL_MID, SOL_WIDTH, VEL_CLIP, EnvConfig,
                        build_geometry, cell_center)
from .pm_common import D as D_DAMP
from .pm_common import G as G_GEAR

import jax
import jax.numpy as jnp

_Q_START_X, _Q_START_Y, _Q_GOAL_X, _Q_GOAL_Y = 0, 1, 2, 3


class EnvState(NamedTuple):
    """All mutable env state. pos/vel/goal [C, N, 2]; step_count/reset_count [C, N]."""
    pos: jnp.ndarray
    vel: jnp.ndarray
    goal: jnp.ndarray
    step_count: jnp.ndarray
    reset_count: jnp.ndarray


def _hash_uniform(key: jnp.ndarray) -> jnp.ndarray:
    """FROZEN RNG (see torch twin): two fmix32 rounds on uint32, uniform = (h >> 8) * 2^-24.

    before: key = uint32 array; after: float32 in [0, 1), bit-identical to the torch draw.
    """
    x = key
    for _ in range(2):
        x = x ^ (x >> 16)
        x = x * jnp.uint32(0x85EBCA6B)
        x = x ^ (x >> 13)
        x = x * jnp.uint32(0xC2B2AE35)
        x = x ^ (x >> 16)
    return (x >> 8).astype(jnp.float32) * jnp.float32(1.0 / 16777216.0)


class JaxPointMaze:
    """Holds the static config/geometry and exposes pure reset/step functions."""

    def __init__(self, cfg: EnvConfig, n_copies: int, n_envs: int, base_seed: int = 0,
                 dtype=jnp.float32, copy_seed_index=None):
        self.cfg, self.C, self.N, self.dtype = cfg, n_copies, n_envs, dtype
        geo = build_geometry(cfg.map_name)
        self.rows, self.cols = geo["rows"], geo["cols"]
        self.nb_mask = jnp.asarray(geo["nb_mask"].reshape(-1).astype(np.int32))
        self.start_center = cell_center(cfg.start_cell, self.rows, self.cols)
        self.goal_center = cell_center(cfg.goal_cell, self.rows, self.cols)
        # per-env identity key (uint32 wrap-around arithmetic = the torch int64-masked math).
        # copy_seed_index lets several copies share one seed stream on purpose: a learning-rate
        # sweep passes the index WITHIN the group, so every group meets the same environments
        # and a difference between groups is the rate's doing. Default: every copy differs.
        # before: copy_seed_index=None -> [0,1,2,...]; 2 groups of 3 -> [0,1,2,0,1,2]
        seed_idx = (jnp.arange(n_copies, dtype=jnp.uint32) if copy_seed_index is None
                    else jnp.asarray(copy_seed_index, dtype=jnp.uint32))
        c = seed_idx[:, None]
        e = jnp.arange(n_envs, dtype=jnp.uint32)[None, :]
        self.id_key = (jnp.uint32(base_seed) * jnp.uint32(0x9E3779B1)
                       + c * jnp.uint32(0x85EBCA77) + e * jnp.uint32(0xC2B2AE3D))

    def _reset_noise(self, reset_count, quantity):
        """One keyed uniform(-noise, +noise) draw per env for the given quantity id."""
        key = (self.id_key + reset_count.astype(jnp.uint32) * jnp.uint32(0x27D4EB2F)
               + jnp.uint32(quantity * 0x165667B1))
        u = _hash_uniform(key).astype(self.dtype)
        return (u * 2.0 - 1.0) * self.cfg.position_noise

    def _spawn(self, reset_count):
        """Fresh (pos, vel, goal) for every env at the given reset generation."""
        pos = jnp.stack([self.start_center[0] + self._reset_noise(reset_count, _Q_START_X),
                         self.start_center[1] + self._reset_noise(reset_count, _Q_START_Y)], -1)
        goal = jnp.stack([self.goal_center[0] + self._reset_noise(reset_count, _Q_GOAL_X),
                          self.goal_center[1] + self._reset_noise(reset_count, _Q_GOAL_Y)], -1)
        return pos, jnp.zeros_like(pos), goal

    def reset(self) -> EnvState:
        """Initial state (generation-0 draws)."""
        rc = jnp.zeros((self.C, self.N), dtype=jnp.int32)
        pos, vel, goal = self._spawn(rc)
        return EnvState(pos, vel, goal, jnp.zeros((self.C, self.N), jnp.int32), rc)

    def dynamics_step(self, pos, vel, act):
        """Pure physics, transliterated from the torch twin (probe-verified exact)."""
        a = jnp.clip(act, -ACT_CLIP, ACT_CLIP)
        v = jnp.clip(vel, -VEL_CLIP, VEL_CLIP)
        F = G_GEAR * a
        a_unc = (F - D_DAMP * v) / M

        j = jnp.clip((pos[..., 0] + self.cols / 2.0).astype(jnp.int32), 0, self.cols - 1)
        i = jnp.clip((self.rows / 2.0 - pos[..., 1]).astype(jnp.int32), 0, self.rows - 1)
        cell = i * self.cols + j
        mask = self.nb_mask[cell]

        px, py = pos[..., 0], pos[..., 1]
        vx_c, vy_c = v[..., 0], v[..., 1]
        aux, auy = a_unc[..., 0], a_unc[..., 1]
        BIG = jnp.asarray(1e9, self.dtype)
        NEGBIG = jnp.asarray(-1e30, self.dtype)
        d1 = jnp.full_like(px, 1e9); b1 = jnp.full_like(px, -1e30)
        R1 = jnp.zeros_like(px); s1x = jnp.zeros_like(px); s1y = jnp.zeros_like(px)
        d2 = jnp.full_like(px, 1e9); b2 = jnp.full_like(px, -1e30)
        R2 = jnp.zeros_like(px); s2x = jnp.zeros_like(px); s2y = jnp.zeros_like(px)
        fi = i.astype(self.dtype)
        fj = j.astype(self.dtype)
        for k, (di, dj) in enumerate(NEIGHBOR_OFFSETS):
            is_wall = (mask & (1 << k)) != 0
            xl = (fj + dj) - self.cols / 2.0
            yb = self.rows / 2.0 - (fi + di + 1.0)
            nx = jnp.clip(px, xl, xl + 1.0)
            ny = jnp.clip(py, yb, yb + 1.0)
            dx, dy = px - nx, py - ny
            dn = jnp.sqrt(jnp.maximum(dx * dx + dy * dy, 1e-24))
            dist = jnp.where(is_wall, dn - R, BIG)
            sx, sy = dx / dn, dy / dn
            active = dist <= MARGIN
            r = dist - MARGIN
            xs = jnp.clip(jnp.abs(r) / SOL_WIDTH, 0.0, 1.0)
            ys = jnp.where(xs < SOL_MID, xs * xs / SOL_MID,
                           1.0 - (1.0 - xs) * (1.0 - xs) / SOL_MID)
            imp = SOL_D0 + (SOL_DMAX - SOL_D0) * ys
            b_qp = (-B_REF * (vx_c * sx + vy_c * sy) - (imp * K_OVER_D) * r
                    - (aux * sx + auy * sy))
            b_qp = jnp.where(active, b_qp, NEGBIG)
            Rreg = (1.0 - imp) / (imp * M)
            take1 = dist < d1
            take2 = (~take1) & (dist < d2)
            d2 = jnp.where(take1, d1, jnp.where(take2, dist, d2))
            b2 = jnp.where(take1, b1, jnp.where(take2, b_qp, b2))
            R2 = jnp.where(take1, R1, jnp.where(take2, Rreg, R2))
            s2x = jnp.where(take1, s1x, jnp.where(take2, sx, s2x))
            s2y = jnp.where(take1, s1y, jnp.where(take2, sy, s2y))
            d1 = jnp.where(take1, dist, d1)
            b1 = jnp.where(take1, b_qp, b1)
            R1 = jnp.where(take1, Rreg, R1)
            s1x = jnp.where(take1, sx, s1x)
            s1y = jnp.where(take1, sy, s1y)

        # exact 2-contact QP by case enumeration (same as the torch twin)
        A11 = 1.0 / M
        A12 = (s1x * s2x + s1y * s2y) / M
        dd1, dd2 = A11 + R1, A11 + R2
        f1_single = jnp.maximum(b1 / dd1, 0.0)
        f2_single = jnp.maximum(b2 / dd2, 0.0)
        det = dd1 * dd2 - A12 * A12
        f1_joint = (dd2 * b1 - A12 * b2) / det
        f2_joint = (dd1 * b2 - A12 * b1) / det
        use_joint = (f1_joint >= 0) & (f2_joint >= 0)
        case1 = (b1 >= 0) & (A12 * f1_single >= b2)
        f1 = jnp.where(use_joint, f1_joint, jnp.where(case1, f1_single, 0.0))
        f2 = jnp.where(use_joint, f2_joint, jnp.where(case1, 0.0, f2_single))

        Fx = F[..., 0] + f1 * s1x + f2 * s2x
        Fy = F[..., 1] + f1 * s1y + f2 * s2y
        v_new = jnp.stack([(M * v[..., 0] + H * Fx), (M * v[..., 1] + H * Fy)], -1) * INV_MHD
        p_new = pos + H * v_new
        return p_new, v_new

    def step(self, state: EnvState, act):
        """One env step with auto-reset. Returns (state', obs, reward, terminated, truncated,
        final_obs) — same contract as the torch twin."""
        pos, vel = self.dynamics_step(state.pos, state.vel, act)
        step_count = state.step_count + 1

        delta = pos - state.goal
        at_goal = (delta * delta).sum(-1) <= self.cfg.goal_radius ** 2
        reward = at_goal.astype(self.dtype) + self.cfg.reward_shift
        if self.cfg.continuing_task:
            terminated = jnp.zeros_like(at_goal)
        else:
            terminated = at_goal
        truncated = (step_count >= self.cfg.max_episode_steps) & ~terminated
        done = terminated | truncated

        final_obs = jnp.concatenate([pos, vel], -1)
        reset_count = state.reset_count + done.astype(jnp.int32)
        s_pos, s_vel, s_goal = self._spawn(reset_count)
        d2 = done[..., None]
        pos = jnp.where(d2, s_pos, pos)
        vel = jnp.where(d2, s_vel, vel)
        goal = jnp.where(d2, s_goal, state.goal)
        step_count = jnp.where(done, 0, step_count)

        obs = jnp.concatenate([pos, vel], -1)
        return (EnvState(pos, vel, goal, step_count, reset_count),
                obs, reward, terminated, truncated, final_obs)


def make_step_batch(dtype: str):
    """Fixture-checker contract: step_batch(pos[B,2], vel[B,2], act[B,2]) -> numpy (pos', vel')."""
    if dtype == "float64":
        jax.config.update("jax_enable_x64", True)
    dt = jnp.float64 if dtype == "float64" else jnp.float32
    env = JaxPointMaze(EnvConfig(), 1, 1, dtype=dt)
    fn = jax.jit(env.dynamics_step)

    def step_batch(pos, vel, act):
        p, v = fn(jnp.asarray(pos, dt), jnp.asarray(vel, dt), jnp.asarray(act, dt))
        return np.asarray(p), np.asarray(v)

    return step_batch
