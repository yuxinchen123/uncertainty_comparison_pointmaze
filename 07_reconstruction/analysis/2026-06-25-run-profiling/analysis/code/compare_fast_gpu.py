#!/usr/bin/env python3
"""
Compare the bold trainer vs SB3 (same node) and CPU vs GPU (same node), from the data/ JSONs.

- fast vs SB3: fast_<algo>_<node>.json (fast_sac_rnd fps) paired with sb3base_<algo>_<node>.json (SB3 fps)
  on the same node -> ratio fast/SB3 per node, averaged.
- CPU vs GPU: gpunode_cpu_<algo>_<node>.json vs gpunode_cuda_<algo>_<node>.json on the same GPU node ->
  ratio cuda/cpu per node.
- Pendulum validation: fast_pendulum_<node>.json mean_return_timed (SAC-core learning sanity).

Usage: python compare_fast_gpu.py
"""
from __future__ import annotations

import glob
import json
import os
import statistics as st

DATA = os.path.join(os.path.dirname(__file__), "..", "..", "data")
ALGOS = ["rnd_state", "gt_position_velocity", "rnd_elliptical"]


def _load(pattern: str) -> dict:
    """{(algorithm, node): fps} for files matching pattern like 'fast_{algo}_*.json'."""
    out = {}
    for algo in ALGOS:
        for f in glob.glob(os.path.join(DATA, pattern.format(algo=algo))):
            try:
                d = json.load(open(f))
            except (json.JSONDecodeError, OSError):
                continue
            out[(algo, d["node"])] = d["fps"]
    return out


def _ratio_table(numer: dict, denom: dict, label: str) -> None:
    """Print per-algorithm same-node ratio numer/denom (paired by node), averaged across nodes."""
    print(f"=== {label} (ratio paired by node) ===")
    for algo in ALGOS:
        ratios, pairs = [], []
        for (a, node), nf in numer.items():
            if a != algo:
                continue
            df = denom.get((algo, node))
            if df and df > 0:
                ratios.append(nf / df)
                pairs.append((node, round(nf, 1), round(df, 1)))
        if ratios:
            print(f"  {algo:22s} x{st.mean(ratios):.3f} ({(st.mean(ratios)-1)*100:+.1f}%) over {len(ratios)} nodes "
                  f"| e.g. {pairs[0]}")


def main():
    fast = _load("fast_{algo}_*.json")
    sb3 = _load("sb3base_{algo}_*.json")
    gcpu = _load("gpunode_cpu_{algo}_*.json")
    gcuda = _load("gpunode_cuda_{algo}_*.json")

    _ratio_table(fast, sb3, "FAST TRAINER vs SB3 (pointmaze, same node)")
    _ratio_table(gcuda, gcpu, "GPU(cuda) vs CPU on the same GPU node")

    # Pendulum SAC-core validation
    print("=== fast_sac_rnd SAC-core validation (Pendulum-v1 mean return; ~-200 = solved) ===")
    for f in sorted(glob.glob(os.path.join(DATA, "fast_pendulum_*.json"))):
        try:
            d = json.load(open(f))
        except (json.JSONDecodeError, OSError):
            continue
        print(f"  {d['node']}: mean_return_timed={d.get('mean_return_timed')} fps={d.get('fps')}")

    # raw fps means
    print("=== raw mean fps ===")
    for label, m in [("fast", fast), ("sb3", sb3), ("gpu_cpu", gcpu), ("gpu_cuda", gcuda)]:
        for algo in ALGOS:
            vals = [v for (a, _), v in m.items() if a == algo]
            if vals:
                print(f"  {label:8s} {algo:22s} fps={st.mean(vals):.2f} (n={len(vals)})")


if __name__ == "__main__":
    main()
