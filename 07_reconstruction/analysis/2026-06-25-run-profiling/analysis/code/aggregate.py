#!/usr/bin/env python3
"""
Aggregate profiling/benchmark JSONs (data/*.json) into per-condition fps stats and same-node speedup ratios.

Each JSON is one run of profile_train.py: {node, condition, algorithm, device, threads, fps, top_tottime, ...}.
This computes, per (algorithm, condition): mean/std fps across nodes; and per algorithm the SAME-NODE ratio
improved_fps / baseline_fps (paired by node, so node-type noise cancels), averaged across nodes.

Usage: python aggregate.py [data_dir] [--baseline baseline]   (prints a table + writes summary.json)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics as st


def load(data_dir: str) -> list:
    """Load every benchmark JSON, tagged by source experiment. before: data/*.json files; after: list[dict]
    each with _source = filename prefix (bench/baseline/fast/sb3base/gpunode/pendulum). The A/B ratio must
    stay within ONE source (the bench runs use 4 cpus/task; the profiling runs use 2 -> not comparable)."""
    recs = []
    for f in glob.glob(os.path.join(data_dir, "*.json")):
        try:
            r = json.load(open(f))
        except (json.JSONDecodeError, OSError):
            continue
        r["_source"] = os.path.basename(f).split("_")[0]
        recs.append(r)
    return recs


def per_condition_fps(recs: list) -> dict:
    """(algorithm, condition) -> {n, mean_fps, std_fps, device}. before: per-run dicts; after: grouped stats."""
    groups = {}
    for r in recs:
        key = (r["algorithm"], r.get("condition", "baseline"), r.get("device", "cpu"))
        groups.setdefault(key, []).append(r["fps"])
    out = {}
    for (algo, cond, dev), fps in sorted(groups.items()):
        out[(algo, cond, dev)] = {
            "n": len(fps), "mean_fps": round(st.mean(fps), 2),
            "std_fps": round(st.pstdev(fps), 2) if len(fps) > 1 else 0.0, "device": dev,
        }
    return out


def same_node_ratios(recs: list, baseline: str) -> dict:
    """(algorithm, condition) -> mean/std of (cond_fps / baseline_fps) paired by node.

    before: per-run dicts with node+condition; after: per (algo,cond) the node-paired speedup ratio stats.
    Pairing by node cancels node-type variation, so the ratio is the clean per-node speedup.
    """
    # index fps by (algo, condition, node)
    idx = {}
    for r in recs:
        idx[(r["algorithm"], r.get("condition", "baseline"), r["node"])] = r["fps"]
    # for each (algo, cond != baseline), collect ratios over nodes that have BOTH cond and baseline
    algos = sorted({r["algorithm"] for r in recs})
    conds = sorted({r.get("condition", "baseline") for r in recs})
    nodes = sorted({r["node"] for r in recs})
    out = {}
    for algo in algos:
        for cond in conds:
            if cond == baseline:
                continue
            ratios = []
            for node in nodes:
                b = idx.get((algo, baseline, node))
                c = idx.get((algo, cond, node))
                if b and c and b > 0:
                    ratios.append(c / b)
            if ratios:
                out[(algo, cond)] = {
                    "n_nodes": len(ratios), "mean_ratio": round(st.mean(ratios), 4),
                    "std_ratio": round(st.pstdev(ratios), 4) if len(ratios) > 1 else 0.0,
                    "pct_speedup": round((st.mean(ratios) - 1) * 100, 2),
                }
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("data_dir", nargs="?", default=os.path.join(os.path.dirname(__file__), "..", "..", "data"))
    p.add_argument("--baseline", default="baseline")
    p.add_argument("--source", default="bench",
                   help="restrict to one experiment source so the A/B is consistent (bench=4cpu sweep); '' = all")
    args = p.parse_args()
    recs = load(args.data_dir)
    if args.source:
        # keep only the matched experiment (e.g. the 'bench' run_bench.slurm runs, all 4 cpus/task)
        recs = [r for r in recs if r.get("_source") == args.source]
    fps = per_condition_fps(recs)
    ratios = same_node_ratios(recs, args.baseline)

    print(f"=== per-condition fps ({len(recs)} runs) ===")
    for (algo, cond, dev), s in fps.items():
        print(f"  {algo:22s} {cond:18s} {dev:4s} n={s['n']:2d} fps={s['mean_fps']:7.2f} +/- {s['std_fps']:.2f}")
    print("=== same-node speedup vs baseline (ratio paired by node) ===")
    for (algo, cond), s in sorted(ratios.items()):
        print(f"  {algo:22s} {cond:18s} x{s['mean_ratio']:.3f} ({s['pct_speedup']:+.1f}%) over {s['n_nodes']} nodes")

    summary = {
        "n_runs": len(recs),
        "per_condition_fps": {f"{a}|{c}|{d}": v for (a, c, d), v in fps.items()},
        "same_node_speedup": {f"{a}|{c}": v for (a, c), v in ratios.items()},
    }
    out_path = os.path.join(os.path.dirname(__file__), "..", "intermediate_results_and_progress", "summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
