#!/usr/bin/env python
"""Invariant checker for the truncation decisions — run every monitoring tick, never trusted away.

The sweep_prune skill requires that any run changing the default rule verify its controller instead
of trusting it. This run changes two things at once (a frozen per-environment bar instead of the
cell's own best mean, and a different score rule per environment), so every decision is re-derived
here from the records on disk and checked against six invariants:

1. no configuration was decided below the seed floor;
2. every recorded truncation satisfied mean + 2.576*s/sqrt(n) < bar at its own n, mean and s;
3. every recorded survivor had at least n_target completed seeds and did NOT satisfy the truncation
   inequality;
4. every decision used the bars file that is on disk now (matching sha256) — the bars are frozen;
5. every decision used its environment's registered score rule;
6. a truncated configuration has no pending markers left, and no configuration is decided twice.

Exit code 0 = all invariants hold; 1 = at least one violation (printed loudly).

Usage:  python truncation_check.py --sweep_id <id> [--n_required 30] [--n_target 100]
"""
import argparse
import hashlib
import json
import glob
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RUN_DIR = os.path.dirname(HERE)
SLURM = os.path.join(RUN_DIR, "slurm")
sys.path.insert(0, SLURM)
import build_queue as bq            # noqa: E402
import score_rules                  # noqa: E402
import truncation_controller as tc  # noqa: E402


def read_decisions(sweep_id):
    """Every decision line of this sweep, in the order it was written."""
    path = os.path.join(SLURM, f"truncation_decisions_{sweep_id}.jsonl")
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as fh:
        for line in fh:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def current_bars_sha():
    """sha256 of the bars file as it stands right now."""
    path = os.path.join(SLURM, "FROZEN_BARS.json")
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def pending_counts(sweep_id):
    """{config_key: number of markers still in pending/} for this sweep."""
    pending = os.path.join(RUN_DIR, "queue", sweep_id, "pending")
    counts = {}
    for path in glob.glob(os.path.join(pending, "*.json")):
        try:
            with open(path) as fh:
                key = json.load(fh).get("config_key")
        except (json.JSONDecodeError, OSError):
            continue
        if key:
            counts[key] = counts.get(key, 0) + 1
    return counts


def check(sweep_id, n_required, n_target):
    """Re-derive every decision and return the list of violation strings (empty = all invariants hold)."""
    decisions = read_decisions(sweep_id)
    violations = []
    sha_now = current_bars_sha()
    seen = {}
    for d in decisions:
        k, verdict, n = d["config_key"], d["verdict"], d["n"]
        upper = d["mean"] + tc.Z99 * d["std"] / math.sqrt(n)
        # 1. seed floor
        if n < n_required:
            violations.append(f"{k}: decided at n={n}, below the floor {n_required}")
        # 2 / 3. the inequality must match the verdict recorded
        if verdict == "truncated" and not upper < d["bar"]:
            violations.append(f"{k}: truncated but upper bound {upper:.4f} >= bar {d['bar']:.4f}")
        if verdict == "survivor":
            if n < n_target:
                violations.append(f"{k}: survivor at n={n}, below the target {n_target}")
            if upper < d["bar"]:
                violations.append(f"{k}: survivor but upper bound {upper:.4f} < bar {d['bar']:.4f}")
        # the recomputed upper bound must also match the one recorded, to the last digits
        if abs(upper - d["upper_99"]) > 1e-9:
            violations.append(f"{k}: recorded upper bound {d['upper_99']} != recomputed {upper}")
        # 4. the bars were frozen: every decision cites the file that is on disk now
        if d.get("bars_sha256") != sha_now:
            violations.append(f"{k}: decided against bars sha256 {d.get('bars_sha256', '')[:16]}, "
                              f"but the file on disk is {sha_now[:16]} — the bars changed mid-run")
        # 5. the environment's registered score rule was used
        expected_rule = score_rules.rule_for(d["env_setup"])
        if d.get("score_rule") != expected_rule:
            violations.append(f"{k}: decided under score rule {d.get('score_rule')!r}, "
                              f"expected {expected_rule!r}")
        # 6. decided at most once
        if k in seen:
            violations.append(f"{k}: decided twice ({seen[k]} then {verdict})")
        seen[k] = verdict
    # 6 (continued). a truncated configuration must have no pending markers left
    counts = pending_counts(sweep_id)
    for k, verdict in seen.items():
        if verdict == "truncated" and counts.get(k, 0) > 0:
            violations.append(f"{k}: truncated but {counts[k]} markers are still pending")
    # every decided key must be a real configuration of this run
    known = {bq.config_key(c) for c in bq.CONFIGS}
    for k in seen:
        if k not in known:
            violations.append(f"{k}: decided, but it is not one of this run's {len(known)} "
                              "configurations")
    return violations, seen


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sweep_id", required=True)
    p.add_argument("--n_required", type=int, default=30)
    p.add_argument("--n_target", type=int, default=100)
    args = p.parse_args()
    violations, seen = check(args.sweep_id, args.n_required, args.n_target)
    n_trunc = sum(1 for v in seen.values() if v == "truncated")
    n_surv = sum(1 for v in seen.values() if v == "survivor")
    print(f"{len(seen)} decided configurations ({n_trunc} truncated, {n_surv} survivors) of "
          f"{len(bq.CONFIGS)}")
    if violations:
        print(f"!!! {len(violations)} INVARIANT VIOLATION(S) !!!")
        for v in violations:
            print(f"  - {v}")
        sys.exit(1)
    print("all truncation invariants hold")


if __name__ == "__main__":
    main()
