#!/usr/bin/env python3
"""
Aggregate the JAX-vs-torch A/B on the exact run-2 task (PointMaze + RND). Pairs sbx vs SB3 by (node, seed)
so node-type noise cancels, over many nodes×seeds for statistics. Reports the full-task speedup (with RND)
and the pure-SAC speedup (extrinsic only).

Usage: python jax_ab_aggregate.py
"""
from __future__ import annotations

import glob
import json
import os
import statistics as st

DATA = os.path.join(os.path.dirname(__file__), "..", "..", "data")


def _load(pattern):
    out = []
    for f in glob.glob(os.path.join(DATA, pattern)):
        try:
            out.append(json.load(open(f)))
        except (json.JSONDecodeError, OSError):
            continue
    return out


def _ab(sb3_recs, sbx_recs, key, label):
    """Paired sbx/sb3 fps ratio over a shared key (e.g. (node,seed)); print mean ±std and raw fps."""
    sb3 = {key(d): d["fps"] for d in sb3_recs}
    sbx = {key(d): d["fps"] for d in sbx_recs}
    pairs = [(sbx[k], sb3[k]) for k in sb3 if k in sbx]
    if not pairs:
        print(f"{label}: (no paired data yet)")
        return
    ratios = [a / b for a, b in pairs]
    print(f"{label}: sbx/SB3 = x{st.mean(ratios):.3f} +/- {st.pstdev(ratios) if len(ratios)>1 else 0:.3f} "
          f"(+{(st.mean(ratios)-1)*100:.0f}%) over {len(ratios)} (node,seed) pairs")
    print(f"   raw fps: SB3 {st.mean([b for _,b in pairs]):.1f} +/-{st.pstdev([b for _,b in pairs]):.1f} | "
          f"sbx {st.mean([a for a,_ in pairs]):.1f} +/-{st.pstdev([a for a,_ in pairs]):.1f}")


def main():
    # full run-2 task: PointMaze + RND (3 seeds/node)
    _ab(_load("jaxab_sb3_rnd_*.json"), _load("jaxab_sbx_rnd_*.json"),
        lambda d: (d["node"], d["seed"]), "FULL TASK (PointMaze + RND)")
    # pure SAC: extrinsic only (1 seed/node)
    _ab(_load("jaxab_sb3_ext_*.json"), _load("jaxab_sbx_ext_*.json"),
        lambda d: d["node"], "PURE SAC (PointMaze, extrinsic)")
    # warmup (JIT) cost for sbx
    sbx = _load("jaxab_sbx_rnd_*.json")
    if sbx:
        print(f"sbx JIT warmup: {st.mean([d['warmup_s'] for d in sbx]):.1f}s (one-time)")


if __name__ == "__main__":
    main()
