"""Batched PointMaze on GPU in pure PyTorch — v0 (correctness first, optimized by the loop).

Two batch axes: n_copies (independent training copies) x n_envs (parallel envs per copy).
All state lives on `device`; step() takes and returns tensors on `device` only — no host sync.
Physics: the probe-verified closed form in ../common/physics_spec.md; walls are hard clamps
(faces per axis, exposed corners radially). Auto-reset: a done env is re-seeded in the same
step; the pre-reset observation is returned separately for correct PPO bootstrapping.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))
from pm_common import (ACT_CLIP, B_REF, H, INV_MHD, K_OVER_D, M, MARGIN, NEIGHBOR_OFFSETS, R,
                       SOL_D0, SOL_DMAX, SOL_MID, SOL_WIDTH, VEL_CLIP, EnvConfig,
                       build_geometry, cell_center)
from pm_common import D as D_DAMP
from pm_common import G as G_GEAR

# quantity ids for the keyed reset-noise RNG (rng-seeding rule: one id per named draw;
# adding a new draw = a new id, existing draws never move)
_Q_START_X, _Q_START_Y, _Q_GOAL_X, _Q_GOAL_Y = 0, 1, 2, 3


def _hash_uniform(key: torch.Tensor) -> torch.Tensor:
    """Counter-based uniform in [0, 1) from an int64 key. FROZEN RNG SPEC (shared by the
    torch / CUDA / JAX implementations so resets are bit-identical across them):
    two murmur3-fmix32 rounds on the 32 low bits, then uniform = (h >> 8) * 2^-24
    (exactly representable in float32, so float32 and float64 envs draw identical noise).

    before: key = int64 tensor mixing (base_seed, copy, env, reset_count, quantity)
    after:  float32 tensor in [0, 1), e.g. key 12345 -> deterministic value
    """
    x = key & 0xFFFFFFFF
    for _ in range(2):
        x = x ^ (x >> 16)
        x = (x * 0x85EBCA6B) & 0xFFFFFFFF
        x = x ^ (x >> 13)
        x = (x * 0xC2B2AE35) & 0xFFFFFFFF
        x = x ^ (x >> 16)
    return (x >> 8).to(torch.float32) * (1.0 / 16777216.0)


class TorchPointMaze:
    """Vectorized PointMaze. Shapes: pos/vel/goal [C, N, 2]; step_count/reset_count [C, N]."""

    def __init__(self, cfg: EnvConfig, n_copies: int, n_envs: int, device="cuda",
                 base_seed: int = 0, dtype=torch.float32, copy_seed_index=None):
        self.cfg, self.C, self.N = cfg, n_copies, n_envs
        self.device, self.dtype = torch.device(device), dtype
        self.base_seed = base_seed

        # geometry lookup: one uint8 neighbor-wall bitmask per cell (bit k = NEIGHBOR_OFFSETS[k]
        # is a wall box); rectangles are reconstructed arithmetically from the cell index
        geo = build_geometry(cfg.map_name)
        self.rows, self.cols = geo["rows"], geo["cols"]
        self.nb_mask = torch.as_tensor(geo["nb_mask"], dtype=torch.int32,
                                       device=self.device).reshape(-1)

        sx, sy = cell_center(cfg.start_cell, self.rows, self.cols)
        gx, gy = cell_center(cfg.goal_cell, self.rows, self.cols)
        self.start_center = torch.tensor([sx, sy], dtype=dtype, device=self.device)
        self.goal_center = torch.tensor([gx, gy], dtype=dtype, device=self.device)

        # per-env identity for the keyed RNG (constant tensors). copy_seed_index lets several
        # copies share one seed stream on purpose: a learning-rate sweep passes the index
        # WITHIN the group, so every group sees the same environments and the comparison
        # between groups is paired. Default is the copy's own index, i.e. all copies differ.
        # before: copy_seed_index=None -> [0, 1, 2, ...]; sweep of 2 groups of 3 -> [0,1,2,0,1,2]
        if copy_seed_index is None:
            seed_idx = torch.arange(n_copies, device=self.device)
        else:
            seed_idx = torch.as_tensor(copy_seed_index, device=self.device).long()
        c_idx = seed_idx.view(-1, 1).expand(n_copies, n_envs)
        e_idx = torch.arange(n_envs, device=self.device).view(1, -1).expand(n_copies, n_envs)
        # base key folds seed/copy/env once; reset_count and quantity are folded per draw
        self._id_key = (base_seed * 0x9E3779B1 + c_idx.long() * 0x85EBCA77
                        + e_idx.long() * 0xC2B2AE3D)

        # mutable state
        z2 = lambda: torch.zeros(n_copies, n_envs, 2, dtype=dtype, device=self.device)
        self.pos, self.vel, self.goal = z2(), z2(), z2()
        self.step_count = torch.zeros(n_copies, n_envs, dtype=torch.int32, device=self.device)
        self.reset_count = torch.zeros(n_copies, n_envs, dtype=torch.int64, device=self.device)

    def _reset_noise(self, reset_count: torch.Tensor, quantity: int) -> torch.Tensor:
        """One keyed uniform(-noise, +noise) draw per env for the given quantity id."""
        key = self._id_key + reset_count * 0x27D4EB2F + quantity * 0x165667B1
        u = _hash_uniform(key).to(self.dtype)
        return (u * 2.0 - 1.0) * self.cfg.position_noise

    def _spawn(self, reset_count: torch.Tensor):
        """Fresh (pos, vel, goal) for every env at the given reset generation.

        before: reset_count [C, N] int64 (how many resets each env has had)
        after:  pos = start center + per-axis noise, vel = 0, goal = goal center + noise
        """
        pos = torch.stack([self.start_center[0] + self._reset_noise(reset_count, _Q_START_X),
                           self.start_center[1] + self._reset_noise(reset_count, _Q_START_Y)], -1)
        goal = torch.stack([self.goal_center[0] + self._reset_noise(reset_count, _Q_GOAL_X),
                            self.goal_center[1] + self._reset_noise(reset_count, _Q_GOAL_Y)], -1)
        vel = torch.zeros_like(pos)
        return pos, vel, goal

    def reset(self) -> torch.Tensor:
        """Reset every env (generation 0 draws); returns obs [C, N, 4]."""
        self.pos, self.vel, self.goal = self._spawn(self.reset_count)
        self.step_count.zero_()
        return torch.cat([self.pos, self.vel], -1)

    def dynamics_step(self, pos, vel, act):
        """Pure physics, probe-verified exact vs MuJoCo: contacts at the CURRENT state produce
        forces (one candidate per neighboring wall box, nearest-point normal, MuJoCo's
        solref/solimp force law, exact 2-contact QP), then one implicit-damping Euler step.
        No position projection — penetration is bounded by the contact dynamics as in MuJoCo.
        Shapes [..., 2]; returns (pos', vel').
        """
        # unconstrained forces: F = G*clip(a) - D*clip(v); a_unc = F/M
        a = act.clamp(-ACT_CLIP, ACT_CLIP)
        v = vel.clamp(-VEL_CLIP, VEL_CLIP)
        F = G_GEAR * a
        a_unc = (F - D_DAMP * v) / M

        # cell of the CURRENT center (never a wall cell: penetration stays a few mm << 0.5)
        # before: pos = world (x, y); after: flat cell index i*cols + j
        j = (pos[..., 0] + self.cols / 2.0).long().clamp(0, self.cols - 1)
        i = (self.rows / 2.0 - pos[..., 1]).long().clamp(0, self.rows - 1)
        cell = i * self.cols + j

        # candidate contacts: 8 neighbor boxes, fully unrolled so the whole pipeline stays one
        # fused elementwise kernel (no topk / gather). Per candidate: reconstruct the neighbor
        # rectangle arithmetically, nearest point -> dist and separation direction s, contact
        # law pieces (b_qp = aref - a_unc_n, Rreg), then a running "two smallest by dist"
        # tournament carrying the payload (b_qp, Rreg, sx, sy).
        mask = self.nb_mask[cell]
        px, py = pos[..., 0], pos[..., 1]
        vx_c, vy_c = v[..., 0], v[..., 1]
        aux, auy = a_unc[..., 0], a_unc[..., 1]
        BIG = torch.full_like(px, 1e9)
        NEGBIG = torch.full_like(px, -1e30)
        ZERO = torch.zeros_like(px)
        d1 = BIG.clone(); b1 = NEGBIG.clone(); R1 = ZERO.clone(); s1x = ZERO.clone(); s1y = ZERO.clone()
        d2 = BIG.clone(); b2 = NEGBIG.clone(); R2 = ZERO.clone(); s2x = ZERO.clone(); s2y = ZERO.clone()
        fi = i.to(pos.dtype)
        fj = j.to(pos.dtype)
        for k, (di, dj) in enumerate(NEIGHBOR_OFFSETS):
            is_wall = (mask & (1 << k)) != 0
            # neighbor box rectangle from the cell index (cells are 1 m, map centered)
            xl = (fj + dj) - self.cols / 2.0
            yb = self.rows / 2.0 - (fi + di + 1.0)
            nx = px.clamp(xl, xl + 1.0)
            ny = py.clamp(yb, yb + 1.0)
            dx, dy = px - nx, py - ny
            dn = (dx * dx + dy * dy).clamp_min(1e-24).sqrt()
            dist = torch.where(is_wall, dn - R, BIG)
            sx, sy = dx / dn, dy / dn
            # contact law pieces (MuJoCo solref/solimp, probe-verified to 8e-16)
            active = dist <= MARGIN
            r = dist - MARGIN
            xs = (r.abs() / SOL_WIDTH).clamp(0.0, 1.0)
            ys = torch.where(xs < SOL_MID, xs * xs / SOL_MID,
                             1.0 - (1.0 - xs) * (1.0 - xs) / SOL_MID)
            imp = SOL_D0 + (SOL_DMAX - SOL_D0) * ys
            b_qp = -B_REF * (vx_c * sx + vy_c * sy) - (imp * K_OVER_D) * r - (aux * sx + auy * sy)
            b_qp = torch.where(active, b_qp, NEGBIG)
            Rreg = (1.0 - imp) / (imp * M)
            # tournament update: candidate k displaces slot 1 or slot 2 by distance
            take1 = dist < d1
            take2 = (~take1) & (dist < d2)
            d2 = torch.where(take1, d1, torch.where(take2, dist, d2))
            b2 = torch.where(take1, b1, torch.where(take2, b_qp, b2))
            R2 = torch.where(take1, R1, torch.where(take2, Rreg, R2))
            s2x = torch.where(take1, s1x, torch.where(take2, sx, s2x))
            s2y = torch.where(take1, s1y, torch.where(take2, sy, s2y))
            d1 = torch.where(take1, dist, d1)
            b1 = torch.where(take1, b_qp, b1)
            R1 = torch.where(take1, Rreg, R1)
            s1x = torch.where(take1, sx, s1x)
            s1y = torch.where(take1, sy, s1y)

        # exact 2-contact QP by case enumeration (KKT of min 1/2 f'(A+R)f - f'b, f >= 0):
        # A11 = A22 = 1/m, A12 = (s1.s2)/m; exactly one case is valid (strictly convex)
        A11 = 1.0 / M
        A12 = (s1x * s2x + s1y * s2y) / M
        d1, d2 = A11 + R1, A11 + R2
        f1_single = (b1 / d1).clamp_min(0.0)
        f2_single = (b2 / d2).clamp_min(0.0)
        det = d1 * d2 - A12 * A12
        f1_joint = (d2 * b1 - A12 * b2) / det
        f2_joint = (d1 * b2 - A12 * b1) / det
        use_joint = (f1_joint >= 0) & (f2_joint >= 0)
        # case {1}: f1 = b1/d1 >= 0 and residual of 2 stays feasible (A12 f1 >= b2)
        case1 = (b1 >= 0) & (A12 * f1_single >= b2)
        f1 = torch.where(use_joint, f1_joint, torch.where(case1, f1_single,
                         torch.zeros_like(b1)))
        f2 = torch.where(use_joint, f2_joint, torch.where(case1, torch.zeros_like(b2),
                         f2_single))

        Fx = F[..., 0] + f1 * s1x + f2 * s2x
        Fy = F[..., 1] + f1 * s1y + f2 * s2y

        # implicit-damping Euler: v' = (M v + h F_total) / (M + h D); q' = q + h v'
        v_new = torch.stack([(M * v[..., 0] + H * Fx), (M * v[..., 1] + H * Fy)], -1) * INV_MHD
        p_new = pos + H * v_new
        return p_new, v_new

    def step_core(self, pos, vel, goal, step_count, reset_count, act):
        """PURE one-step transition (no attribute writes) — usable inside CUDA-graph capture.

        Inputs are the five state tensors plus act [C, N, 2]. Returns
        (pos', vel', goal', step_count', reset_count', obs, reward, terminated, truncated,
        final_obs) with the same semantics as step().
        """
        pos, vel = self.dynamics_step(pos, vel, act)
        step_count = step_count + 1

        # reward and episode ends on the post-step position
        delta = pos - goal
        at_goal = (delta * delta).sum(-1) <= self.cfg.goal_radius ** 2
        reward = at_goal.to(self.dtype) + self.cfg.reward_shift
        if self.cfg.continuing_task:
            terminated = torch.zeros_like(at_goal)
        else:
            terminated = at_goal
        truncated = (step_count >= self.cfg.max_episode_steps) & ~terminated
        done = terminated | truncated

        final_obs = torch.cat([pos, vel], -1)

        # auto-reset the finished envs (branch-free: fresh spawn computed for all, masked in)
        reset_count = reset_count + done.long()
        s_pos, s_vel, s_goal = self._spawn(reset_count)
        d2 = done.unsqueeze(-1)
        pos = torch.where(d2, s_pos, pos)
        vel = torch.where(d2, s_vel, vel)
        goal = torch.where(d2, s_goal, goal)
        step_count = torch.where(done, torch.zeros_like(step_count), step_count)

        obs = torch.cat([pos, vel], -1)
        return pos, vel, goal, step_count, reset_count, obs, reward, terminated, truncated, final_obs

    def step(self, act: torch.Tensor):
        """One env step for every copy/env (stateful wrapper around step_core). act [C, N, 2].

        Returns (obs [C,N,4], reward [C,N], terminated [C,N] bool, truncated [C,N] bool,
        final_obs [C,N,4]) — obs is post-auto-reset; final_obs is the pre-reset observation
        (equal to obs where the env did not finish), for value bootstrapping.
        """
        (self.pos, self.vel, self.goal, self.step_count, self.reset_count,
         obs, reward, terminated, truncated, final_obs) = self.step_core(
            self.pos, self.vel, self.goal, self.step_count, self.reset_count, act)
        return obs, reward, terminated, truncated, final_obs
