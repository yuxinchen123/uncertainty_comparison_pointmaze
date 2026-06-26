#!/usr/bin/env python3
"""
Local (per-function) microbenchmarks for the bit-exact switches, isolated from the training loop:
- polyak target update: SB3 zip_strict loop vs torch._foreach_, on the SAC twin-critic parameter shapes.
- intrinsic reward combine: numpy round-trip vs torch-only, on a (256,1) reward batch.
These are the LOCAL improvements (the whole-loop fps is the end-to-end). Relative timing is robust to
node contention because each is a tight pure-compute loop.

Usage: python micro_local.py
"""
from __future__ import annotations

import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/p/rlprojects/RND/07_reconstruction")
import stable_baselines3.common.utils as U  # noqa: E402
from rnd_exploration.common.sb3_patches import polyak_update_foreach  # noqa: E402
from rnd_exploration.common.format import to_numpy_flat, to_tensor  # noqa: E402

torch.set_num_threads(2)


def _time(fn, n):
    """Median-ish wall time per call over n iterations (after a short warmup)."""
    for _ in range(20):
        fn()
    t0 = time.time()
    for _ in range(n):
        fn()
    return (time.time() - t0) / n * 1e6  # microseconds/call


def bench_polyak():
    """Time the soft target update: SB3 zip_strict loop vs torch._foreach_ (same 12 critic-param shapes x2)."""
    shapes = [(256, 4), (256,), (256, 256), (256,), (1, 256), (1,)] * 2
    params = [torch.randn(s) for s in shapes]
    tgt = [torch.randn(s) for s in shapes]
    zip_us = _time(lambda: U.polyak_update(params, tgt, 0.005), 3000)
    fe_us = _time(lambda: polyak_update_foreach(params, tgt, 0.005), 3000)
    print(f"polyak update:    zip_strict {zip_us:6.1f} us/call  vs  foreach {fe_us:6.1f} us/call  "
          f"-> local x{zip_us/fe_us:.2f} ({(zip_us/fe_us-1)*100:+.0f}%)")


def bench_reward_combine():
    """Time the reward combine: numpy round-trip vs torch-only, on a (256,1) batch."""
    device = torch.device("cpu")
    batch_r = torch.randn(256, 1)
    intrinsic = torch.randn(256)
    beta = 100.0

    def numpy_path():
        i = to_numpy_flat(intrinsic); e = to_numpy_flat(batch_r)
        total = (e + beta * i).reshape(-1, 1)
        return to_tensor(total, device)

    def torch_path():
        it = intrinsic.reshape(-1, 1).to(dtype=batch_r.dtype, device=batch_r.device)
        return batch_r + beta * it

    np_us = _time(numpy_path, 5000)
    th_us = _time(torch_path, 5000)
    print(f"reward combine:   numpy r-trip {np_us:6.1f} us/call  vs  torch {th_us:6.1f} us/call  "
          f"-> local x{np_us/th_us:.2f} ({(np_us/th_us-1)*100:+.0f}%)")


if __name__ == "__main__":
    bench_polyak()
    bench_reward_combine()
