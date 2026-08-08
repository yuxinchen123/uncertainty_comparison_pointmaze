#!/usr/bin/env python
"""One-pass extraction: every per-run JSON of runs 1.1 / 1.2 / the lr addendum -> one compact
summary line per record in data/summaries.jsonl.gz.

Each record (300 kB - several MB) is reduced to a few kB: per-50k-step-bin episode counts,
success counts and reward sums; the raw step and length of every SUCCESSFUL episode; the
coverage trajectory; and whole-run aggregates. All figures and tables of this analysis read
only the cache, so the 12.8 GB of records is parsed exactly once.

Records are parsed with json.load, never a head/tail window (the head+tail reader silently
returns mid-training values on large records — bug found and fixed 2026-08-06).

Usage:
  extract_summaries.py [--sources run11,run12,addendum] [--jobs 8] [--limit N] [--force]

Shards are written per source (data/shard_<source>.jsonl.gz) and merged; a re-run only redoes
missing shards, so the extraction is resumable.
"""
import argparse
import glob
import gzip
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ANA = os.path.dirname(HERE)
TRAIN_RUNS = os.path.dirname(os.path.dirname(os.path.dirname(ANA)))

BIN = 50_000   # the accumulation grid; same cadence train_history/eval_history already use


def _first(pattern):
    """The single path matching a glob, or a hard error (no silent fallback)."""
    hits = sorted(glob.glob(pattern))
    if len(hits) != 1:
        sys.exit(f"expected exactly one match for {pattern}, got {len(hits)}")
    return hits[0]


SOURCES = {
    "run11": os.path.join(_first(os.path.join(TRAIN_RUNS, "*run_8_1_pointmaze-antmaze-8env*")),
                          "data", "2026-07-23-02-05_pm-am-run1", "local"),
    "run12": os.path.join(_first(os.path.join(TRAIN_RUNS, "*run_8_1_2_antmaze-umaze-medium*")),
                          "data", "2026-08-01-02-03_run812", "local"),
    "addendum": os.path.join(_first(os.path.join(TRAIN_RUNS, "*run_5_8_1_2_addendum*")),
                             "data", "2026-08-05-16-05_lr1e3", "local"),
}


def arm_of(d):
    """A record's arm name across the three sweeps' conventions.

    before: run-1.1 record with algorithm="gt_position_1m"        -> "gt_position_1m"
            run-1.2 record with rnd_optimizer="adam"              -> "baseline"
            run-1.2 record with sgd + rnd_layer_norm=True         -> "alg2.3"
            addendum record with adam + rnd_lr=0.01               -> "origsmall_lr0.01"
    """
    algo = d.get("algorithm", "?")
    if algo != "rnd_next_state":
        return algo                       # the run-1.1 oracle / no-exploration arms
    if d.get("_source") == "addendum":
        return f'origsmall_lr{"%g" % float(d.get("rnd_lr", 0))}'
    if d.get("rnd_optimizer") == "adam":
        return "baseline"                 # run-1.1 RND and run-1.2 task R
    # run-1.2 task-S arms, the inverse of that run's ARM_EXTRAS (same logic as its build_queue)
    if str(d.get("rnd_layer_norm")) == "True":
        return "alg2.3"
    if d.get("rnd_predictor_loss") == "mse_init_normalized":
        return "alg2.2"
    if str(d.get("rnd_readout_norm_init")) == "True":
        return "alg2.1"
    return "alg1"


def summarize(path, source):
    """One record -> one summary dict, or None when unusable (not completed / no episodes)."""
    try:
        with open(path) as fh:
            d = json.load(fh)
    except (ValueError, OSError):
        return None
    if not d.get("completed"):
        return None
    eps = d.get("train_episode_history") or []
    if not eps:
        return None
    d["_source"] = source
    total = int(d.get("total_timesteps", 0))
    nbins = max(1, total // BIN)

    # per-bin accumulators over the episode stream
    # before: 14370 rows {"step": 700, "train/extrinsic_reward": -700.0, "train/success": false, ...}
    # after : n_ep=[71,72,...], n_succ=[0,3,...], sum_ext=[-49700.0,...], sum_int=[...], per 50k bin
    n_ep = [0] * nbins
    n_succ = [0] * nbins
    sum_ext = [0.0] * nbins
    sum_int = [0.0] * nbins
    succ_steps, succ_lens = [], []
    ident_viol = 0
    ext_all = []
    for r in eps:
        b = min(int(r["step"]) // BIN, nbins - 1)
        ext = float(r["train/extrinsic_reward"])
        length = int(r["train/episode_length"])
        succ = bool(r.get("train/success", False))
        n_ep[b] += 1
        sum_ext[b] += ext
        sum_int[b] += abs(float(r.get("train/intrinsic_reward", 0.0)))
        ext_all.append(ext)
        # the reward identity the credit-assignment math relies on: -(length-1) on success,
        # -length on a timeout (V5 of the plan)
        if succ:
            n_succ[b] += 1
            succ_steps.append(int(r["step"]))
            succ_lens.append(length)
            if abs(ext + (length - 1)) > 0.51:
                ident_viol += 1
        elif abs(ext + length) > 0.51:
            ident_viol += 1

    last100 = ext_all[-100:]
    succ_flags = [bool(r.get("train/success", False)) for r in eps]
    ev = d.get("eval_history") or []
    return {
        "source": source, "file": os.path.basename(path),
        "env_setup": d.get("env_setup"), "arm": arm_of(d),
        "beta": "%g" % float(d.get("beta", 0)),
        "rnd_lr": "%g" % float(d.get("rnd_lr", 0)) if d.get("rnd_lr") is not None else "-",
        "reward_norm": str(d.get("rnd_reward_norm")) == "True",
        "total_timesteps": total, "a_seed": d.get("a_seed"),
        "n_episodes": len(eps),
        "R_whole": sum(ext_all) / len(ext_all),
        "R_last100": sum(last100) / len(last100),
        "succ_whole": sum(succ_flags) / len(succ_flags),
        "succ_last100": sum(succ_flags[-100:]) / len(succ_flags[-100:]),
        "first_success_step": succ_steps[0] if succ_steps else None,
        "last_success_step": succ_steps[-1] if succ_steps else None,
        "episode_cap": max(int(r["train/episode_length"]) for r in eps),
        "n_ep": n_ep, "n_succ": n_succ, "sum_ext": sum_ext, "sum_int": sum_int,
        "succ_steps": succ_steps, "succ_lens": succ_lens,
        "eval_steps": [int(r["step"]) for r in ev],
        "cov_cell": [r.get("visit_counts/coverage_pct") for r in ev],
        "cov_1m": [r.get("visit_counts_1m/coverage_pct") for r in ev],
        "ident_viol": ident_viol,
    }


def extract_source(source, jobs, limit):
    """Extract one source directory into its shard; returns (kept, seen)."""
    shard = os.path.join(ANA, "data", f"shard_{source}.jsonl.gz")
    paths = sorted(glob.glob(os.path.join(SOURCES[source], "*.json")))
    if limit:
        paths = paths[:limit]
    kept = 0
    with gzip.open(shard, "wt") as out:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            for s in pool.map(summarize, paths, [source] * len(paths), chunksize=16):
                if s is not None:
                    out.write(json.dumps(s) + "\n")
                    kept += 1
    return kept, len(paths)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sources", default="run11,run12,addendum")
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--limit", type=int, default=0, help="files per source (smoke test)")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    os.makedirs(os.path.join(ANA, "data"), exist_ok=True)
    manifest = {}
    for source in args.sources.split(","):
        shard = os.path.join(ANA, "data", f"shard_{source}.jsonl.gz")
        if os.path.exists(shard) and not args.force:
            print(f"[{source}] shard exists, skipping (--force to redo)")
            continue
        kept, seen = extract_source(source, args.jobs, args.limit)
        manifest[source] = {"seen": seen, "kept_completed": kept}
        print(f"[{source}] {kept} summaries from {seen} files")
    # merge shards into the single cache every downstream reader uses
    merged = os.path.join(ANA, "data", "summaries.jsonl.gz")
    with gzip.open(merged, "wt") as out:
        for source in SOURCES:
            shard = os.path.join(ANA, "data", f"shard_{source}.jsonl.gz")
            if os.path.exists(shard):
                with gzip.open(shard, "rt") as fh:
                    for line in fh:
                        out.write(line)
    with open(os.path.join(ANA, "data", "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)
    print(f"merged cache: {merged}")


if __name__ == "__main__":
    main()
