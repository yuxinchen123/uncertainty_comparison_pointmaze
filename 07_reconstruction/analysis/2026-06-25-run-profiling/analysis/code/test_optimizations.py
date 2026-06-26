#!/usr/bin/env python3
"""
Correctness tests for the performance switches added to the main codebase (each optimization must produce
the SAME result as the baseline, or a documented numerically-close one). Run:

    conda run -n exploration python -m pytest test_optimizations.py -q

These test the main-repo switches (currently: opt_torch_reward in VectorIntrinsicReplayBuffer). When an
optimization is accepted, its test moves into the main repo's tests/ alongside the code.
"""
from __future__ import annotations

import sys
import numpy as np
import torch
from gymnasium import spaces

sys.path.insert(0, "/p/rlprojects/RND/07_reconstruction")
from rnd_exploration.buffers.vector_intrinsic_replay_buffer import VectorIntrinsicReplayBuffer  # noqa: E402


class _MockIntrinsic:
    """Deterministic intrinsic model: compute() = per-row obs sum (stable); update() is a no-op (no state drift)."""

    def compute(self, samples):
        # a deterministic per-sample scalar so both buffer paths see identical intrinsic values
        return samples["observations"].sum(dim=1)

    def update(self, samples):
        # no-op: keeps the model state fixed so sampling twice is comparable
        pass


def _make_filled_buffer(opt_torch_reward: bool, beta: float = 2.0) -> VectorIntrinsicReplayBuffer:
    """Build a small buffer with the given switch, filled with 100 fixed-seed random transitions."""
    obs_space = spaces.Box(-1.0, 1.0, (4,), dtype=np.float32)
    act_space = spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)
    buf = VectorIntrinsicReplayBuffer(
        200, obs_space, act_space, device="cpu",
        beta=beta, intrinsic_reward_model=_MockIntrinsic(), opt_torch_reward=opt_torch_reward,
    )
    # identical data in both buffers (same RandomState seed) so only the reward-combine path differs
    rng = np.random.RandomState(0)
    for _ in range(100):
        buf.add(
            rng.randn(1, 4).astype(np.float32), rng.randn(1, 4).astype(np.float32),
            rng.randn(1, 2).astype(np.float32), rng.randn(1).astype(np.float32),
            np.array([False]), [{}],
        )
    return buf


def test_torch_reward_matches_numpy_reward_bit_identical():
    """opt_torch_reward ON vs OFF produce bit-identical rewards (and the SB3 (batch,1) shape) on the same batch."""
    buf_off = _make_filled_buffer(opt_torch_reward=False)
    buf_on = _make_filled_buffer(opt_torch_reward=True)
    # seed numpy so both sample() calls draw the SAME batch_inds (sample() uses np.random.randint)
    np.random.seed(123)
    s_off = buf_off.sample(32)
    np.random.seed(123)
    s_on = buf_on.sample(32)
    assert s_off.rewards.shape == (32, 1) and s_on.rewards.shape == (32, 1)
    assert s_off.rewards.dtype == s_on.rewards.dtype
    # float32 ext + beta*intrinsic is the same IEEE computation in numpy and torch -> exactly equal
    assert torch.equal(s_off.rewards, s_on.rewards), \
        f"max abs diff {torch.max(torch.abs(s_off.rewards - s_on.rewards)).item()}"


def test_polyak_foreach_bit_exact():
    """torch._foreach_ polyak update is bit-identical to SB3's zip_strict loop (same ops, same element order)."""
    from stable_baselines3.common.utils import polyak_update as sb3_polyak
    from rnd_exploration.common.sb3_patches import polyak_update_foreach
    torch.manual_seed(0)
    # SAC twin-critic parameter shapes (two critics -> the shape list twice)
    shapes = [(256, 4), (256,), (256, 256), (256,), (1, 256), (1,)] * 2
    params = [torch.randn(s) for s in shapes]
    tgt_orig = [torch.randn(s) for s in shapes]
    tgt_sb3 = [t.clone() for t in tgt_orig]
    tgt_fe = [t.clone() for t in tgt_orig]
    tau = 0.005
    sb3_polyak(params, tgt_sb3, tau)            # SB3 reference loop
    polyak_update_foreach(params, tgt_fe, tau)  # the foreach replacement
    assert all(torch.equal(a, b) for a, b in zip(tgt_sb3, tgt_fe)), "foreach polyak diverged from SB3"
    # empty-list guard: SAC calls polyak on the (empty for MlpPolicy) batch_norm_stats list -> must no-op
    polyak_update_foreach([], [], tau)


if __name__ == "__main__":
    test_torch_reward_matches_numpy_reward_bit_identical()
    test_polyak_foreach_bit_exact()
    print("ok")
