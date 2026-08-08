#!/usr/bin/env python
"""Shared loader + metric helpers for the non-convergence analysis. Everything reads the
one-pass cache; nothing here opens a raw per-run record."""
import gzip
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ANA = os.path.dirname(HERE)
PLOTS = os.path.join(ANA, "plots")
CACHE = os.path.join(ANA, "data", "summaries.jsonl.gz")

BIN = 50_000

UMAZE = "AntMaze_UMaze-v5_start_bottom_left"
MEDIUM = "AntMaze_Medium-v5_start_bottom_left"
ENV_SHORT = {UMAZE: "AntMaze UMaze", MEDIUM: "AntMaze Medium"}
CAP = {UMAZE: 700, MEDIUM: 1000}
GAMMA = 0.99


def load(**filters):
    """Cache rows matching all keyword filters (value or list of values).

    before: load(source="run12", arm="baseline", total_timesteps=10_000_000)
    after : [summary dict, ...] — only the 10M task-R baseline rows
    """
    rows = []
    with gzip.open(CACHE, "rt") as fh:
        for line in fh:
            d = json.loads(line)
            ok = True
            for k, v in filters.items():
                want = v if isinstance(v, (list, tuple, set)) else (v,)
                if d.get(k) not in want:
                    ok = False
                    break
            if ok:
                rows.append(d)
    return rows


def mean_se(vals):
    """Sample mean and standard error of a non-empty list; (nan, nan) when empty."""
    n = len(vals)
    if n == 0:
        return float("nan"), float("nan")
    m = sum(vals) / n
    if n < 2:
        return m, 0.0
    sd = math.sqrt(sum((x - m) ** 2 for x in vals) / (n - 1))
    return m, sd / math.sqrt(n)


def bin_success(row, agg=1):
    """A row's per-bin success rate on the 50k grid, aggregated by `agg` bins.

    before: n_ep=[71,72,...200 bins...], n_succ=[0,3,...], agg=20 (a 1M-step display bin)
    after : ([1e6, 2e6, ...], [0.033, 0.059, ...]) — rate per aggregated bin
    """
    n_ep, n_succ = row["n_ep"], row["n_succ"]
    steps, rates = [], []
    for i in range(0, len(n_ep), agg):
        e = sum(n_ep[i:i + agg])
        s = sum(n_succ[i:i + agg])
        steps.append((i + agg) * BIN)
        rates.append(s / e if e else float("nan"))
    return steps, rates


def delta_gamma(k, cap, gamma=GAMMA):
    """The discounted start-state advantage of succeeding at step k over timing out at cap."""
    return (gamma ** k - gamma ** cap) / (1 - gamma)


def best_config(rows, key=lambda r: r["R_whole"], min_seeds=5):
    """(beta, rows-of-that-beta) with the best mean of `key`, over configs with >= min_seeds."""
    by_beta = {}
    for r in rows:
        by_beta.setdefault(r["beta"], []).append(r)
    ranked = [(b, sum(key(r) for r in v) / len(v)) for b, v in by_beta.items()
              if len(v) >= min_seeds]
    if not ranked:
        return None, []
    beta = max(ranked, key=lambda t: t[1])[0]
    return beta, by_beta[beta]
