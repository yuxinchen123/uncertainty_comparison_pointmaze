#!/usr/bin/env python
"""Report Train-run-3.1.1 sweep progress: finished seeds per config, and whether every one of the 91
configs has reached the >=30-seed threshold that triggers the writeup update.

Usage: progress.py <sweep_id>   (reads data/<sweep_id>/local/*.json under the run folder)

A "config" is one of the 91 per-seed points: for the elliptical methods the key is
(algorithm, update_timing, feature_input, regularization, beta); for RND it is (algorithm, beta). The
finished count per config is the number of distinct a_seed values whose run JSON exists.
"""
import os
import sys
import json
import glob
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
RUN_DIR = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from build_queue import build_configs, RUN_TOTAL  # noqa: E402

THRESHOLD = 30


def config_key_from_spec(spec):
    """Canonical key for an expected config spec (from build_configs): elliptical -> full key, RND -> (algo,beta)."""
    algo = spec["algorithm"]
    beta = float(spec["beta"])
    p = spec["params"]
    if p:
        return (algo, p["elliptical_update_timing"], p["elliptical_feature_input"],
                float(p["elliptical_regularization"]), beta)
    return (algo, None, None, None, beta)


def config_key_from_record(r):
    """Canonical key for a finished run JSON (same shape as config_key_from_spec)."""
    algo = r["algorithm"]
    beta = float(r["beta"])
    if "elliptical_update_timing" in r:  # elliptical runs carry these; RND runs omit them
        return (algo, r["elliptical_update_timing"], r["elliptical_feature_input"],
                float(r["elliptical_regularization"]), beta)
    return (algo, None, None, None, beta)


def main():
    """Count finished distinct seeds per config and print a summary + THRESHOLD_MET / THRESHOLD_NOT_MET line."""
    sweep_id = sys.argv[1]
    expected = {config_key_from_spec(s) for s in build_configs()}
    # before: empty; after: per-config set of finished seeds
    seeds_by_cfg = defaultdict(set)
    data_dir = os.path.join(RUN_DIR, "data", sweep_id, "local")
    files = glob.glob(os.path.join(data_dir, "*.json"))
    n_bad = 0
    n_partial = 0
    for f in files:
        try:
            r = json.load(open(f))
        except (json.JSONDecodeError, OSError):
            n_bad += 1  # a file mid-write; skip this cycle
            continue
        # checkpoint records (completed=false, written at eval cadence since 2026-07-02) are partial runs,
        # not finished ones; a missing flag means an old write-once record, i.e. complete.
        if r.get("completed", True) is False:
            n_partial += 1
            continue
        seeds_by_cfg[config_key_from_record(r)].add(int(r["a_seed"]))
    # per-config finished counts over ALL expected configs (missing config -> 0)
    counts = {k: len(seeds_by_cfg.get(k, set())) for k in expected}
    total_done = sum(counts.values())
    ge30 = sum(1 for c in counts.values() if c >= THRESHOLD)
    mn = min(counts.values()) if counts else 0
    ordered = sorted(counts.values())
    med = ordered[len(ordered) // 2] if ordered else 0
    # per-algorithm breakdown: configs at >=30 out of that algorithm's config count
    by_algo = defaultdict(lambda: [0, 0])  # algo -> [configs_ge30, n_configs]
    for k, c in counts.items():
        by_algo[k[0]][1] += 1
        if c >= THRESHOLD:
            by_algo[k[0]][0] += 1
    print(f"sweep={sweep_id} total_done={total_done}/{RUN_TOTAL} files={len(files)} bad_read={n_bad} partial={n_partial}")
    print(f"configs={len(expected)} configs_ge{THRESHOLD}={ge30}/{len(expected)} min={mn} median={med}")
    for algo in sorted(by_algo):
        g, n = by_algo[algo]
        print(f"  {algo:24s} configs>= {THRESHOLD}: {g}/{n}")
    # the trigger: every config must have >= THRESHOLD finished seeds
    if mn >= THRESHOLD:
        print(f"THRESHOLD_MET min={mn}")
    else:
        # name a few of the most-behind configs to see what is lagging
        behind = sorted(counts.items(), key=lambda kv: kv[1])[:3]
        bs = "; ".join(f"{k}={c}" for k, c in behind)
        print(f"THRESHOLD_NOT_MET min={mn} behind: {bs}")


if __name__ == "__main__":
    main()
