#!/usr/bin/env python
"""Two-phase truncation controller for train run 6 — this run's ONE controller.

Rules (the user's instruction for this run):

- Score: one number per completed record, train run 5's rule (slurm/score_rules.py): the final
  training-episode reward.
- PHASE 1 — TRUNCATE against the frozen bar, from 20 seeds. A configuration c with
  n_c >= n_required (20) completed seeds is truncated at the first cycle where its one-sided 99%
  upper confidence limit falls below the frozen bar,

      mean_c + 2.576 * s_c / sqrt(n_c)  <  BAR,

  where BAR is the run-5 original RND best configuration's mean (38.6412, Adam 1e-4, bonus weight
  1000, n=300), read from slurm/FROZEN_BARS.json (committed; its sha256 is recorded on every
  decision line and verified by 20_mins_monitoring/truncation_check.py). Re-checked every cycle as
  n grows toward 100.
- PHASE 2 — WINNER PER ARM, at 100 seeds. When a configuration reaches n_race (100) completed
  seeds without a phase-1 truncation, it becomes a phase-2 CANDIDATE. Once every still-undecided
  configuration of an arm is resolved (truncated, or a candidate), the arm's candidate with the
  best mean CONTINUES to n_final (300) seeds — verdict `winner`; every other candidate of that arm
  is stopped — verdict `stopped_at_100`. An arm whose configurations were all truncated in phase 1
  has no winner. A winner reaching n_final completed seeds gets the final verdict `winner_complete`.
- The controller NEVER submits jobs and NEVER adds queue entries. It only MOVES pending markers of
  a decided-away configuration into queue/<sweep_id>/pruned/ (all remaining markers for
  `truncated`; the seed-100-and-above markers for `stopped_at_100`) and appends one decision line.
  Seeds already running always finish and are kept.

Restart-safe: decided configurations are rebuilt each cycle from the decision log; the marker move
precedes the log line and re-moving an already-empty configuration is a no-op, so a crash between
the two repeats the idempotent side effect and logs exactly once.

Usage:  python truncation_controller.py --sweep_id <id> [--n_required 20] [--n_race 100]
        [--n_final 300] [--once]
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
import build_queue   # noqa: E402  (CONFIGS / config_key / label / key_from_record / ARMS)
import score_rules   # noqa: E402  (the score rule)

Z99 = 2.576   # the 99% one-sided normal quantile of the sweep_prune rule
STEPS = build_queue.STEPS


def bars():
    """The frozen bar {env: mean} plus the file's sha256. Hard-fails when the file is missing —
    the controller must never invent a bar."""
    path = os.path.join(HERE, "FROZEN_BARS.json")
    if not os.path.exists(path):
        sys.exit(f"missing {path} — run compute_frozen_bars.py once at launch and commit it")
    raw = open(path, "rb").read()
    doc = json.loads(raw)
    return {env: v["mean"] for env, v in doc["bars"].items()}, hashlib.sha256(raw).hexdigest()


def load_scores(sweep_id):
    """Return {config_key: [score, ...]} over this sweep's COMPLETED per-run JSONs.

    Memory-bounded: each record is reduced to one float here and dropped.
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
    """{config_key: verdict} from the decision log (the controller's restart-safe state).

    `candidate` lines mark phase-2 entry and are superseded by `winner` / `stopped_at_100`;
    `winner` is superseded by `winner_complete`. Later lines win, so replaying the log in order
    reproduces the state.
    """
    path = os.path.join(HERE, f"truncation_decisions_{sweep_id}.jsonl")
    decided = {}
    if os.path.exists(path):
        with open(path) as fh:
            for line in fh:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if d.get("verdict") in ("truncated", "candidate", "winner", "stopped_at_100",
                                        "winner_complete"):
                    decided[d["config_key"]] = d["verdict"]
    return decided


def move_pending(sweep_id, cfg_key, min_seed_index=0):
    """Move pending markers of cfg_key with seed_index >= min_seed_index into pruned/. Returns the
    count moved. Idempotent: a second call finds nothing left to move.

    min_seed_index=0 removes every remaining marker (phase-1 truncation); min_seed_index=100
    removes only the not-yet-needed tail (phase-2 stop at 100).
    """
    spec = next(c for c in build_queue.CONFIGS if build_queue.config_key(c) == cfg_key)
    tag = build_queue.label(spec)
    pending = os.path.join(RUN_DIR, "queue", sweep_id, "pending")
    pruned = os.path.join(RUN_DIR, "queue", sweep_id, "pruned")
    os.makedirs(pruned, exist_ok=True)
    moved = 0
    for path in glob.glob(os.path.join(pending, f"*_of_*_{tag}_seed*.json")):
        if min_seed_index > 0:
            # the marker's seed index is recoverable from its run id: id = seed_index*len(CONFIGS)+i
            run_id = int(os.path.basename(path).split("_", 1)[0])
            if run_id // len(build_queue.CONFIGS) < min_seed_index:
                continue
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


def run_cycle(sweep_id, n_required, n_race, n_final):
    """One controller pass: phase-1 truncations, phase-2 candidacies, per-arm winner decisions,
    and winner completion."""
    BARS, bars_sha = bars()
    scores = load_scores(sweep_id)
    decided = already_decided(sweep_id)
    summary = []

    def common_of(cfg_spec, k, vals):
        m, s = mean_std(vals)
        n = len(vals)
        return {"config_key": k, "env_setup": cfg_spec["env_setup"], "arm": cfg_spec["arm"],
                "score_rule": score_rules.rule_for(cfg_spec["env_setup"]),
                "n": n, "mean": m, "std": s,
                "upper_99": m + Z99 * s / math.sqrt(n) if n else float("nan"),
                "bar": BARS[cfg_spec["env_setup"]], "bars_sha256": bars_sha}

    # phase 1 + phase-2 candidacy, per configuration
    for cfg_spec in build_queue.CONFIGS:
        k = build_queue.config_key(cfg_spec)
        verdict = decided.get(k)
        if verdict in ("truncated", "winner", "stopped_at_100", "winner_complete"):
            continue
        vals = scores.get(k, [])
        n = len(vals)
        if n < n_required:
            continue
        common = common_of(cfg_spec, k, vals)
        if common["upper_99"] < common["bar"]:
            # side effects FIRST (idempotent), log line LAST -> crash-safe, never double-acts
            moved = move_pending(sweep_id, k)
            decided[k] = "truncated"
            log_decision(sweep_id, dict(common, verdict="truncated", pending_moved=moved,
                                        n_required=n_required))
            summary.append(f"truncated {k}: n={n} upper={common['upper_99']:.4f} < bar "
                           f"{common['bar']:.4f}")
        elif n >= n_race and verdict != "candidate":
            decided[k] = "candidate"
            log_decision(sweep_id, dict(common, verdict="candidate", n_race=n_race))
            summary.append(f"candidate {k}: n={n} mean={common['mean']:.4f}")

    # phase 2: an arm whose every configuration is resolved (truncated or candidate) crowns its
    # best-mean candidate; the other candidates stop at 100
    for arm in build_queue.ARMS:
        arm_keys = [build_queue.config_key(c) for c in build_queue.CONFIGS if c["arm"] == arm]
        if any(decided.get(k) in (None,) for k in arm_keys):
            continue  # some configuration still racing toward 20/100 — the arm is not resolved
        if any(decided.get(k) in ("winner", "winner_complete") for k in arm_keys):
            pass      # winner already chosen; fall through to completion check
        else:
            cands = [k for k in arm_keys if decided.get(k) == "candidate"]
            if not cands:
                continue  # every configuration truncated: the arm has no winner
            best = max(cands, key=lambda k: mean_std(scores.get(k, [0.0]))[0])
            for k in cands:
                cfg_spec = next(c for c in build_queue.CONFIGS if build_queue.config_key(c) == k)
                common = common_of(cfg_spec, k, scores.get(k, []))
                if k == best:
                    decided[k] = "winner"
                    log_decision(sweep_id, dict(common, verdict="winner", n_final=n_final))
                    summary.append(f"winner {k}: n={common['n']} mean={common['mean']:.4f} "
                                   f"-> continues to {n_final} seeds")
                else:
                    moved = move_pending(sweep_id, k, min_seed_index=n_race)
                    decided[k] = "stopped_at_100"
                    log_decision(sweep_id, dict(common, verdict="stopped_at_100",
                                                pending_moved=moved, n_race=n_race))
                    summary.append(f"stopped_at_100 {k}: n={common['n']} "
                                   f"mean={common['mean']:.4f}")
        # winner completion: the final verdict once the winner has its 300 seeds
        for k in arm_keys:
            if decided.get(k) == "winner" and len(scores.get(k, [])) >= n_final:
                cfg_spec = next(c for c in build_queue.CONFIGS if build_queue.config_key(c) == k)
                common = common_of(cfg_spec, k, scores.get(k, []))
                decided[k] = "winner_complete"
                log_decision(sweep_id, dict(common, verdict="winner_complete", n_final=n_final))
                summary.append(f"winner_complete {k}: n={common['n']} mean={common['mean']:.4f}")
    return summary


def main():
    if getpass.getuser() != "sl5nw":
        sys.exit("owner-only script; collaborators never decide")
    p = argparse.ArgumentParser()
    p.add_argument("--sweep_id", required=True)
    p.add_argument("--n_required", type=int, default=20,
                   help="completed-seed floor before any phase-1 decision (this run: 20)")
    p.add_argument("--n_race", type=int, default=100,
                   help="phase-1 race target; a survivor becomes a phase-2 candidate here")
    p.add_argument("--n_final", type=int, default=300,
                   help="the per-arm winner's final seed count")
    p.add_argument("--once", action="store_true")
    p.add_argument("--poll_seconds", type=int, default=1200)
    args = p.parse_args()
    while True:
        lines = run_cycle(args.sweep_id, args.n_required, args.n_race, args.n_final)
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
