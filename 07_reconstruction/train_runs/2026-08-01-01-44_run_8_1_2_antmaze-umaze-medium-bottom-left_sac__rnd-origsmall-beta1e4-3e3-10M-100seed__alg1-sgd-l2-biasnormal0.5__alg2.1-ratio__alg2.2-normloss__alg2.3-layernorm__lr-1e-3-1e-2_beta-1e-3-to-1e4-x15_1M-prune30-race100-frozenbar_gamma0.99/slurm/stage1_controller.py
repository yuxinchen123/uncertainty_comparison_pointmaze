#!/usr/bin/env python
"""Stage-1 frozen-bar controller for train run 8.1.2 — the run's ONE central controller.

Rules (this run's design; adapted from run 8.1's prune_controller.py, cell-relative bar replaced by
the FROZEN bar, plus a final survivor verdict):
- Score = the whole-run mean per-episode extrinsic return of a completed TASK-S (1M-step) record.
  Task-R (baseline, 10M) records never enter a decision: the baseline arm is exempt.
- PRUNE: config c with n_c >= n_required completed 1M seeds is pruned at the first cycle where
  mean_c + 2.576 * s_c / sqrt(n_c) < BAR(env) — the frozen run-8.1 RND winner mean for that env,
  read from slurm/FROZEN_BARS.json (committed; its sha256 is logged with every decision and
  verified by 20_mins_monitoring/stage1_check.py). Re-checked every cycle as n grows toward 100.
- SURVIVOR: a config reaching >= n_target (100) completed seeds without ever failing the check gets
  a final "survivor" verdict — the input list for the deferred stage 2.
- The controller NEVER submits jobs and NEVER adds queue entries — it only MOVES pending_1m markers
  to queue/<sweep_id>/pruned/ and records decisions. Already-running seeds always finish.

Restart-safe: decided configs are rebuilt each cycle from slurm/stage1_decisions_<sweep_id>.jsonl;
marker moves precede the log line and re-moving an already-empty config is a no-op, so a crash
between the two re-runs the idempotent side effect and logs exactly once.

Usage:  python stage1_controller.py --sweep_id <id> [--n_required 30] [--n_target 100] [--once]
"""
import argparse
import getpass
import glob
import hashlib
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RUN_DIR = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import build_queue  # noqa: E402  (CONFIGS / config_key / label / key_from_record: source of truth)

Z99 = 2.576   # the 99% normal value of the sweep_prune skill's upper-bound rule
STEPS_S = build_queue.STEPS_S


def bars():
    """The frozen bars {env: mean} plus the file's sha256. Hard-fails when the file is missing —
    the controller must never invent a bar."""
    path = os.path.join(HERE, "FROZEN_BARS.json")
    if not os.path.exists(path):
        sys.exit(f"missing {path} — run compute_frozen_bar.py once at launch and commit it")
    raw = open(path, "rb").read()
    doc = json.loads(raw)
    return {env: v["mean"] for env, v in doc["bars"].items()}, hashlib.sha256(raw).hexdigest()


def score_of_record(d):
    """The record's racing score: mean per-episode extrinsic return over ALL training episodes
    (identical to run 8.1's rule and to compute_frozen_bar.score_of_record)."""
    rows = d.get("train_episode_history") or []
    vals = [r["train/extrinsic_reward"] for r in rows if "train/extrinsic_reward" in r]
    if not vals:
        return None
    return sum(vals) / len(vals)


def load_scores(sweep_id):
    """Return {config_key: [score, ...]} over COMPLETED task-S per-run JSONs of this sweep.
    Task-R records (total_timesteps 10M / arm baseline) are filtered out — the baseline is exempt.
    Memory-bounded: each record reduces to one float here and is dropped."""
    local = os.path.join(RUN_DIR, "data", sweep_id, "local")
    scores = {}
    for path in glob.glob(os.path.join(local, "*.json")):
        try:
            with open(path) as fh:
                d = json.load(fh)
        except (json.JSONDecodeError, OSError):
            continue  # a record mid-flush this cycle; picked up next cycle
        if not d.get("completed", True):
            continue  # partial checkpoint of a killed attempt: never enters a racing mean
        if int(d.get("total_timesteps", 0)) != STEPS_S:
            continue  # task-R baseline record (10M): exempt from every decision
        key = build_queue.key_from_record(d)
        if "|baseline|" in key:
            continue  # defense in depth: an adam-stack record can never be decided on
        s = score_of_record(d)
        if s is None:
            continue
        scores.setdefault(key, []).append(s)
    return scores


def mean_std(vals):
    """Sample mean and (n-1) standard deviation of a non-empty list."""
    n = len(vals)
    mean = sum(vals) / n
    if n < 2:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in vals) / (n - 1)
    return mean, math.sqrt(var)


def already_decided(sweep_id):
    """{config_key: verdict} from the decision log (restart-safe state)."""
    path = os.path.join(HERE, f"stage1_decisions_{sweep_id}.jsonl")
    decided = {}
    if os.path.exists(path):
        with open(path) as fh:
            for line in fh:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if d.get("verdict") in ("pruned", "survivor"):
                    decided[d["config_key"]] = d["verdict"]
    return decided


def move_pending(sweep_id, cfg_key):
    """Move every pending_1m marker of cfg_key to pruned/. Returns the count moved. Idempotent:
    a second call finds nothing to move."""
    spec = next(c for c in build_queue.CONFIGS if build_queue.config_key(c) == cfg_key)
    tag = build_queue.label(spec)
    pending = os.path.join(RUN_DIR, "queue", sweep_id, "pending_1m")
    pruned = os.path.join(RUN_DIR, "queue", sweep_id, "pruned")
    os.makedirs(pruned, exist_ok=True)
    moved = 0
    for path in glob.glob(os.path.join(pending, f"*_of_*_{tag}_seed*.json")):
        try:
            os.rename(path, os.path.join(pruned, os.path.basename(path)))
            moved += 1
        except OSError:
            pass  # a worker claimed it in this instant; that one extra seed just runs
    return moved


def log_decision(sweep_id, record):
    """Append one decision line to the decisions JSONL (flushed immediately)."""
    path = os.path.join(HERE, f"stage1_decisions_{sweep_id}.jsonl")
    record["time"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(path, "a") as fh:
        fh.write(json.dumps(record) + "\n")
        fh.flush()


def run_cycle(sweep_id, n_required, n_target):
    """One controller pass: score every undecided task-S config, prune vs the frozen bar, declare
    survivors at the seed target."""
    BARS, bars_sha = bars()
    scores = load_scores(sweep_id)
    decided = already_decided(sweep_id)
    summary = []
    for cfg_spec in build_queue.CONFIGS:
        k = build_queue.config_key(cfg_spec)
        if k in decided:
            continue
        vals = scores.get(k, [])
        n = len(vals)
        if n < n_required:
            continue
        m, s = mean_std(vals)
        upper = m + Z99 * s / math.sqrt(n)
        bar = BARS[cfg_spec["env_setup"]]
        if upper < bar:
            # side effects FIRST (idempotent), log line LAST -> crash-safe, never double-acts
            moved = move_pending(sweep_id, k)
            decided[k] = "pruned"
            log_decision(sweep_id, {
                "verdict": "pruned", "config_key": k, "env_setup": cfg_spec["env_setup"],
                "n": n, "mean": m, "std": s, "upper_99": upper,
                "bar": bar, "bars_sha256": bars_sha,
                "pending_moved": moved, "n_required": n_required,
            })
            summary.append(f"pruned {k}: n={n} mean={m:.2f} upper={upper:.2f} < bar {bar:.2f}")
        elif n >= n_target:
            # final verdict: all seeds completed and the config never failed the check
            decided[k] = "survivor"
            log_decision(sweep_id, {
                "verdict": "survivor", "config_key": k, "env_setup": cfg_spec["env_setup"],
                "n": n, "mean": m, "std": s, "upper_99": upper,
                "bar": bar, "bars_sha256": bars_sha, "n_target": n_target,
            })
            summary.append(f"survivor {k}: n={n} mean={m:.2f} upper={upper:.2f} >= bar {bar:.2f}")
    return summary


def main():
    if getpass.getuser() != "sl5nw":
        sys.exit("owner-only script; collaborators never decide")
    p = argparse.ArgumentParser()
    p.add_argument("--sweep_id", required=True)
    p.add_argument("--n_required", type=int, default=30,
                   help="seed floor before any decision (this run: 30, the skill default)")
    p.add_argument("--n_target", type=int, default=100, help="race target = the survivor threshold")
    p.add_argument("--once", action="store_true")
    p.add_argument("--poll_seconds", type=int, default=1200)
    args = p.parse_args()
    while True:
        lines = run_cycle(args.sweep_id, args.n_required, args.n_target)
        stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
        for ln in lines:
            print(f"[{stamp}] {ln}", flush=True)
        if not lines:
            print(f"[{stamp}] no new stage-1 decisions", flush=True)
        if args.once:
            break
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
