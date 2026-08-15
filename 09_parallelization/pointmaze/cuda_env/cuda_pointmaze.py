"""Fused-CUDA PointMaze wrapper — same API and exact same numbers as TorchPointMaze.

The whole env step runs in one kernel (pointmaze_kernel.cu), one thread per env; float32
math is bit-identical to the torch eager twin (compiled with --fmad=false, same expression
grouping, same frozen fmix32 reset RNG). Output tensors are PREALLOCATED and reused by the
next step call — callers that keep a step's outputs must copy them.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_HOME", "/localtmp/sl5nw/cuda_home")
os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/localtmp/sl5nw/torch_ext")
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0")

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))
from pm_common import EnvConfig, build_geometry, cell_center  # noqa: E402

_EXT = None


def _ext():
    """Compile-on-first-use handle to the CUDA extension (cached in TORCH_EXTENSIONS_DIR).

    PM_TOURNAMENT selects which form of the two-nearest-wall selection is compiled in:
    "count" (default) ranks the eight distances by counting, "select" carries the winners
    through a running two-slot tournament. They compute the same thing; each gets its own
    extension name so both can be loaded and compared in one benchmark session.
    """
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load
        form = os.environ.get("PM_TOURNAMENT", "count")
        flags = ["-O3", "--fmad=false"]
        if form == "count":
            flags.append("-DPM_RANK_TOURNAMENT")
        elif form != "select":
            raise ValueError(f"PM_TOURNAMENT must be 'count' or 'select', got {form!r}")
        _EXT = load(name=f"pointmaze_cuda_ext_{form}",
                    sources=[str(Path(__file__).resolve().parent / "pointmaze_kernel.cu")],
                    extra_cuda_cflags=flags, verbose=False)
    return _EXT


class CudaPointMaze:
    """Vectorized PointMaze, fused CUDA backend. Mirrors TorchPointMaze's API and shapes."""

    def __init__(self, cfg: EnvConfig, n_copies: int, n_envs: int, device="cuda",
                 base_seed: int = 0, dtype=torch.float32):
        self.cfg, self.C, self.N = cfg, n_copies, n_envs
        self.device, self.dtype = torch.device(device), dtype
        self.base_seed = base_seed
        self.block = 256

        geo = build_geometry(cfg.map_name)
        self.rows, self.cols = geo["rows"], geo["cols"]
        self.nb_mask = torch.as_tensor(geo["nb_mask"].astype(np.int32),
                                       device=self.device).reshape(-1).contiguous()
        self.start_center = cell_center(cfg.start_cell, self.rows, self.cols)
        self.goal_center = cell_center(cfg.goal_cell, self.rows, self.cols)

        # packed state [C, N, 4] = (x, y, vx, vy); pos/vel are stride-4 views of it and the
        # post-reset state IS the returned observation (no separate obs buffer or write)
        self.state = torch.zeros(n_copies, n_envs, 4, dtype=dtype, device=self.device)
        self.pos = self.state[..., 0:2]
        self.vel = self.state[..., 2:4]
        self.goal = torch.zeros(n_copies, n_envs, 2, dtype=dtype, device=self.device)
        self.step_count = torch.zeros(n_copies, n_envs, dtype=torch.int32, device=self.device)
        self.reset_count = torch.zeros(n_copies, n_envs, dtype=torch.int32, device=self.device)

        # preallocated per-step outputs (reused every call)
        self._reward = torch.zeros(n_copies, n_envs, dtype=dtype, device=self.device)
        self._terminated = torch.zeros(n_copies, n_envs, dtype=torch.bool, device=self.device)
        self._truncated = torch.zeros(n_copies, n_envs, dtype=torch.bool, device=self.device)
        self._final_obs = torch.zeros(n_copies, n_envs, 4, dtype=dtype, device=self.device)

    def reset(self) -> torch.Tensor:
        """Reset every env at its current reset generation; returns obs [C, N, 4]."""
        _ext().env_reset(self.state, self.goal, self.step_count, self.reset_count,
                         self.N, self.start_center[0], self.start_center[1],
                         self.goal_center[0], self.goal_center[1], self.cfg.position_noise,
                         self.base_seed, self.block)
        return self.state

    def step(self, act: torch.Tensor):
        """One fused env step. act [C, N, 2] on device, env dtype, contiguous."""
        _ext().env_step(self.state, self.goal, self.step_count, self.reset_count,
                        act.contiguous(), self._reward, self._terminated,
                        self._truncated, self._final_obs, self.nb_mask,
                        self.N, self.rows, self.cols,
                        self.start_center[0], self.start_center[1],
                        self.goal_center[0], self.goal_center[1],
                        self.cfg.position_noise, self.cfg.goal_radius ** 2,
                        self.cfg.reward_shift, self.cfg.max_episode_steps,
                        self.cfg.continuing_task, self.base_seed, self.block)
        return self.state, self._reward, self._terminated, self._truncated, self._final_obs

    def dynamics_step(self, pos, vel, act):
        """Pure physics on arbitrary [..., 2] tensors (fixture-checker path)."""
        p_in = pos.reshape(-1, 2).contiguous()
        v_in = vel.reshape(-1, 2).contiguous()
        a_in = act.reshape(-1, 2).contiguous()
        p_out = torch.empty_like(p_in)
        v_out = torch.empty_like(v_in)
        _ext().env_dynamics(p_in, v_in, a_in, p_out, v_out, self.nb_mask,
                            self.rows, self.cols, self.block)
        return p_out.view_as(pos), v_out.view_as(vel)


def make_step_batch(dtype: str):
    """Fixture-checker contract: step_batch(pos[B,2], vel[B,2], act[B,2]) -> numpy (pos', vel')."""
    dt = torch.float64 if dtype == "float64" else torch.float32
    env = CudaPointMaze(EnvConfig(), 1, 1, device="cuda", dtype=dt)

    def step_batch(pos, vel, act):
        p = torch.tensor(pos, dtype=dt, device="cuda")
        v = torch.tensor(vel, dtype=dt, device="cuda")
        a = torch.tensor(act, dtype=dt, device="cuda")
        p2, v2 = env.dynamics_step(p, v, a)
        return p2.cpu().numpy(), v2.cpu().numpy()

    return step_batch
