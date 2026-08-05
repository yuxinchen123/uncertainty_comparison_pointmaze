#!/usr/bin/env python
"""Truncation controller for the Adam learning-rate 1e-3 addendum — this run's ONE controller.

Rules (the user's instruction for this run; the frozen-bar shape of train run 1.2, with train run
5's score rule on the PointMaze environment — see slurm/score_rules.py):

- Score: one number per completed record, under ITS OWN environment's rule. PointMaze uses train
  run 5's final reward; the two AntMaze environments use train run 1.1/1.2's whole-run mean.
- TRUNCATE: configuration c with n_c >= n_required (30) completed seeds is truncated at the first
  cycle where its one-sided 99% upper confidence limit falls below its environment's frozen bar,

      mean_c + 2.576 * s_c / sqrt(n_c)  <  BAR(env)

  where BAR(env) is the mean of the SAME environment's best Adam 1e-4 configuration, read from
  slurm/FROZEN_BARS.json (committed; its sha256 is recorded on every decision line and verified by
  20_mins_monitoring/truncation_check.py). Re-checked every cycle as n grows toward 100.
- SURVIVOR: a configuration reaching n_target (100) completed seeds without ever failing the check
  gets a final "survivor" verdict.
- The controller NEVER submits jobs and NEVER adds queue entries. It only MOVES pending markers of a
  truncated configuration into queue/<sweep_id>/pruned/ and appends one decision line. Seeds already
  running always finish and are kept.

Restart-safe: decided configurations are rebuilt each cycle from the decision log; the marker move
precedes the log line and re-moving an already-empty configuration is a no-op, so a crash between
the two repeats the idempotent side effect and logs exactly once.

Usage:  python truncation_controller.py --sweep_id <id> [--n_required 30] [--n_target 100] [--once]
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
import build_queue   # noqa: E402  (CONFIGS / config_key / label / key_from_record)
import score_rules   # noqa: E402  (the per-environment score rule)

Z99 = 2.576   # the 99% one-sided normal quantile of the sweep_prune rule
STEPS = build_queue.STEPS


def bars():
    """The frozen bars {env: mean} plus the file's sha256. Hard-fails when the file is missing — the
    controller must never invent a bar."""
    path = os.path.join(HERE, "FROZEN_BARS.json")
    if not os.path.exists(path):
        sys.exit(f"missing {path} — run compute_frozen_bars.py once at launch and commit it")
    raw = open(path, "rb").read()
    doc = json.loads(raw)
    return {env: v["mean"] for env, v in doc["bars"].items()}, hashlib.sha256(raw).hexdigest()


def load_scores(sweep_id):
    """Return {config_key: [score, ...]} over this sweep's COMPLETED per-run JSONs.

    Memory-bounded: each record is reduced to one float here and dropped, so a 4,500-record sweep
    costs a few kilobytes of state.
    """
    local = os.path.join(RUN_DIR, "data", sweep_id, "local")
    scores = {}
    for path in glob.glob(os.path.join(local, "*.json")):
        try:
            with open(path) as fh:
                d = json.load(fh)
        except (json.JSONDecodeError, OSError):
            continue  # a record mid-flush this cycle; picked up next cycle
        if not d.get("completed", True):
            continue  # partial checkpoint of a killed attempt never enters a racing mean
        if int(d.get("total_timesteps", 0)) != STEPS:
            continue  # defense in depth: only this run's 1M-step records are decided on
        s = score_rules.score_of_record(d)
        if s is None:
            continue
        scores.setdefault(build_queue.key_from_record(d), []).append(s)
    return scores


def mean_std(vals):
    """Sample mean and (n-1) standard deviation of a non-empty list."""
    n = len(vals)
    mean = sum(vals) / n
    if n < 2:
        return mean, 0.0
    return mean, math.sqrt(sum((v - mean) ** 2 for v in vals) / (n - 1))


def already_decided(sweep_id):
    """{config_key: verdict} from the decision log (the controller's restart-safe state)."""
    path = os.path.join(HERE, f"truncation_decisions_{sweep_id}.jsonl")
    decided = {}
    if os.path.exists(path):
        with open(path) as fh:
            for line in fh:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if d.get("verdict") in ("truncated", "survivor"):
                    decided[d["config_key"]] = d["verdict"]
    return decided


def move_pending(sweep_id, cfg_key):
    """Move every pending marker of cfg_key into pruned/. Returns the count moved. Idempotent: a
    second call finds nothing left to move."""
    spec = next(c for c in build_queue.CONFIGS if build_queue.config_key(c) == cfg_key)
    tag = build_queue.label(spec)
    pending = os.path.join(RUN_DIR, "queue", sweep_id, "pending")
    pruned = os.path.join(RUN_DIR, "queue", sweep_id, "pruned")
    os.makedirs(pruned, exist_ok=True)
    moved = 0
    for path in glob.glob(os.path.join(pending, f"*_of_*_{tag}_seed*.json")):
        try:
            os.rename(path, os.path.join(pruned, os.path.basename(path)))
            moved += 1
        except OSError:
            pass  # a worker claimed it in this instant; that one extra seed simply runs
    return moved


def log_decision(sweep_id, record):
    """Append one decision line to the decisions JSONL, flushed immediately."""
    path = os.path.join(HERE, f"truncation_decisions_{sweep_id}.jsonl")
    record["time"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(path, "a") as fh:
        fh.write(json.dumps(record) + "\n")
        fh.flush()


def run_cycle(sweep_id, n_required, n_target):
    """One controller pass: score every undecided configuration, truncate against its environment's
    frozen bar, declare survivors at the seed target."""
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
        env = cfg_spec["env_setup"]
        m, s = mean_std(vals)
        upper = m + Z99 * s / math.sqrt(n)
        bar = BARS[env]
        common = {"config_key": k, "env_setup": env, "score_rule": score_rules.rule_for(env),
                  "n": n, "mean": m, "std": s, "upper_99": upper,
                  "bar": bar, "bars_sha256": bars_sha}
        if upper < bar:
            # side effects FIRST (idempotent), log line LAST -> crash-safe, never double-acts
            moved = move_pending(sweep_id, k)
            decided[k] = "truncated"
            log_decision(sweep_id, dict(common, verdict="truncated", pending_moved=moved,
                                        n_required=n_required))
            summary.append(f"truncated {k}: n={n} mean={m:.4f} upper={upper:.4f} < bar {bar:.4f}")
        elif n >= n_target:
            decided[k] = "survivor"
            log_decision(sweep_id, dict(common, verdict="survivor", n_target=n_target))
            summary.append(f"survivor {k}: n={n} mean={m:.4f} upper={upper:.4f} >= bar {bar:.4f}")
    return summary


def main():
    if getpass.getuser() != "sl5nw":
        sys.exit("owner-only script; collaborators never decide")
    p = argparse.ArgumentParser()
    p.add_argument("--sweep_id", required=True)
    p.add_argument("--n_required", type=int, default=30,
                   help="completed-seed floor before any decision (this run: 30)")
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
            print(f"[{stamp}] no new truncation decisions", flush=True)
        if args.once:
            break
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
