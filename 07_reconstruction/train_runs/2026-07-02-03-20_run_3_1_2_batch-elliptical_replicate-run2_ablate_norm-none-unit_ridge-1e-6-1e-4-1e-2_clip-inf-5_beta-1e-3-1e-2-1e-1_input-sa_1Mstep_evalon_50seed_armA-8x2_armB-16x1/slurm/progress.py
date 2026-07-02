#!/usr/bin/env python
"""Report Train-run-3.1.2 progress across BOTH arms: completed seeds per config (pooled + per arm),
partial-checkpoint counts, stale running/ markers, and whether every one of the 36 configs has reached
the >=20 pooled-seed threshold that triggers the writeup update.

Usage: progress.py <sweep_id_A> <sweep_id_B>

A "config" is one of the 36 cells x beta points, keyed by
(normalization, ridge, clip, beta) — all read from each finished JSON's own fields. Only records with
completed=true count as finished (checkpoint records with completed=false are partial). A running/ marker
older than PER_RUN_TIMEOUT (24 h) counts as a stale/lost run for the infra metrics.
"""
import os
import sys
import json
import glob
import time
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
RUN_DIR = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from build_queue import build_configs  # noqa: E402

THRESHOLD = 20
STALE_SECONDS = 24 * 60 * 60  # matches worker.py PER_RUN_TIMEOUT


def config_key_from_spec(spec):
    """Canonical key for an expected config: (normalization, ridge as float, clip as float, beta as float)."""
    p = spec["params"]
    return (p["elliptical_feature_normalization"], float(p["elliptical_regularization"]),
            float(p["elliptical_bonus_clip"]), float(spec["beta"]))


def config_key_from_record(r):
    """Canonical key for a finished run JSON (same shape as config_key_from_spec)."""
    return (r["elliptical_feature_normalization"], float(r["elliptical_regularization"]),
            float(r["elliptical_bonus_clip"]), float(r["beta"]))


def scan_arm(sweep_id):
    """One arm's counts: completed seeds per config, partial/bad file counts, queue-state tallies."""
    seeds_by_cfg = defaultdict(set)
    n_bad = n_partial = 0
    for f in glob.glob(os.path.join(RUN_DIR, "data", sweep_id, "local", "*.json")):
        try:
            r = json.load(open(f))
        except (json.JSONDecodeError, OSError):
            n_bad += 1
            continue
        # checkpoint records are unfinished runs; only completed=true counts as done
        if r.get("completed", True) is False:
            n_partial += 1
            continue
        seeds_by_cfg[config_key_from_record(r)].add(int(r["a_seed"]))
    # queue-state tallies incl. stale running/ markers (job killed at walltime leaves the marker forever)
    q = os.path.join(RUN_DIR, "queue", sweep_id)
    counts = {s: len(os.listdir(os.path.join(q, s))) if os.path.isdir(os.path.join(q, s)) else 0
              for s in ("pending", "running", "done", "failed")}
    now = time.time()
    stale = sum(1 for p in glob.glob(os.path.join(q, "running", "*.json"))
                if now - os.path.getmtime(p) > STALE_SECONDS)
    return seeds_by_cfg, n_bad, n_partial, counts, stale


def main():
    """Print per-arm and pooled coverage plus the THRESHOLD_MET / THRESHOLD_NOT_MET trigger line."""
    sweep_a, sweep_b = sys.argv[1], sys.argv[2]
    expected = {config_key_from_spec(s) for s in build_configs()}
    per_arm = {}
    pooled = defaultdict(set)
    for arm, sid in (("A", sweep_a), ("B", sweep_b)):
        seeds_by_cfg, n_bad, n_partial, counts, stale = scan_arm(sid)
        per_arm[arm] = (seeds_by_cfg, n_bad, n_partial, counts, stale)
        for k, seeds in seeds_by_cfg.items():
            pooled[k] |= seeds
        done_runs = sum(len(v) for v in seeds_by_cfg.values())
        print(f"arm {arm} ({sid}): completed={done_runs}/900 partial={n_partial} bad={n_bad} "
              f"queue(p/r/d/f)={counts['pending']}/{counts['running']}/{counts['done']}/{counts['failed']} "
              f"stale_running={stale}")
    # pooled coverage over the 36 expected configs (missing config -> 0)
    counts_pooled = {k: len(pooled.get(k, set())) for k in expected}
    mn = min(counts_pooled.values())
    ge = sum(1 for c in counts_pooled.values() if c >= THRESHOLD)
    ordered = sorted(counts_pooled.values())
    print(f"pooled: configs_ge{THRESHOLD}={ge}/{len(expected)} min={mn} median={ordered[len(ordered)//2]}")
    if mn >= THRESHOLD:
        print(f"THRESHOLD_MET min={mn}")
    else:
        behind = sorted(counts_pooled.items(), key=lambda kv: kv[1])[:3]
        print("THRESHOLD_NOT_MET min=%d behind: %s" % (mn, "; ".join(f"{k}={c}" for k, c in behind)))


if __name__ == "__main__":
    main()
