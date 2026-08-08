#!/usr/bin/env python
"""Invariant checker for train run 6's two-phase truncation decisions — run every monitoring tick,
never trusted away.

The sweep_prune skill requires that any run changing the default rule verify its controller
instead of trusting it. This run changes the rule twice over (a frozen external bar with a 20-seed
floor, then a per-arm winner selection at 100 seeds), so every decision is re-derived here from
the decision log and checked against these invariants:

1.  no phase-1 decision below the 20-seed floor; no candidacy below 100 seeds; no winner
    completion below 300;
2.  every recorded truncation satisfied mean + 2.576*s/sqrt(n) < bar at its own n, mean and s;
    every recorded candidate did NOT satisfy it;
3.  per arm, at most ONE configuration holds a winner/winner_complete verdict, and its recorded
    mean is >= every stopped_at_100 sibling's recorded mean;
4.  a winner exists for an arm only once every configuration of that arm is decided
    (truncated / candidate-then-resolved), never while a sibling is still racing;
5.  every decision used the bars file on disk now (matching sha256) — the bars are frozen;
6.  every decision used the environment's registered score rule;
7.  a truncated configuration has no pending markers left; a stopped_at_100 configuration has no
    pending markers at seed index >= 100; verdicts follow the legal chains
    (truncated | candidate -> winner -> winner_complete | candidate -> stopped_at_100);
    every decided key is a real configuration.

Exit code 0 = all invariants hold; 1 = at least one violation (printed loudly).

Usage:  python truncation_check.py --sweep_id <id> [--n_required 20] [--n_race 100] [--n_final 300]
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

# verdict -> the verdicts allowed to FOLLOW it in the log (None = terminal)
LEGAL_NEXT = {
    "truncated": set(),
    "candidate": {"winner", "stopped_at_100"},
    "winner": {"winner_complete"},
    "stopped_at_100": set(),
    "winner_complete": set(),
}


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


def pending_by_key_and_seed(sweep_id):
    """{config_key: [seed_index, ...]} of the markers still in pending/ for this sweep."""
    pending = os.path.join(RUN_DIR, "queue", sweep_id, "pending")
    out = {}
    for path in glob.glob(os.path.join(pending, "*.json")):
        try:
            with open(path) as fh:
                d = json.load(fh)
        except (json.JSONDecodeError, OSError):
            continue
        if d.get("config_key"):
            out.setdefault(d["config_key"], []).append(int(d.get("seed_index", 0)))
    return out


def check(sweep_id, n_required, n_race, n_final):
    """Re-derive every decision; return (violations, final_verdict_by_key)."""
    decisions = read_decisions(sweep_id)
    violations = []
    sha_now = current_bars_sha()
    state = {}          # config_key -> latest verdict
    winner_mean = {}    # arm -> the winner's recorded mean at decision time
    stopped_means = {}  # arm -> [stopped_at_100 recorded means]
    for d in decisions:
        k, verdict, n = d["config_key"], d["verdict"], d["n"]
        upper = d["mean"] + tc.Z99 * d["std"] / math.sqrt(n) if n else float("nan")
        arm = d.get("arm")
        # 1. floors per verdict kind
        if verdict == "truncated" and n < n_required:
            violations.append(f"{k}: truncated at n={n}, below the floor {n_required}")
        if verdict in ("candidate", "winner", "stopped_at_100") and n < n_race:
            violations.append(f"{k}: {verdict} at n={n}, below the race target {n_race}")
        if verdict == "winner_complete" and n < n_final:
            violations.append(f"{k}: winner_complete at n={n}, below the final target {n_final}")
        # 2. the phase-1 inequality must match the verdict
        if verdict == "truncated" and not upper < d["bar"]:
            violations.append(f"{k}: truncated but upper bound {upper:.4f} >= bar {d['bar']:.4f}")
        if verdict == "candidate" and upper < d["bar"]:
            violations.append(f"{k}: candidate but upper bound {upper:.4f} < bar {d['bar']:.4f}")
        if abs(upper - d["upper_99"]) > 1e-9:
            violations.append(f"{k}: recorded upper bound {d['upper_99']} != recomputed {upper}")
        # 5. frozen bars
        if d.get("bars_sha256") != sha_now:
            violations.append(f"{k}: decided against bars sha256 {d.get('bars_sha256', '')[:16]}, "
                              f"but the file on disk is {sha_now[:16]} — the bars changed mid-run")
        # 6. registered score rule
        expected_rule = score_rules.rule_for(d["env_setup"])
        if d.get("score_rule") != expected_rule:
            violations.append(f"{k}: decided under score rule {d.get('score_rule')!r}, "
                              f"expected {expected_rule!r}")
        # 7. legal verdict chains
        prev = state.get(k)
        if prev is not None and verdict not in LEGAL_NEXT.get(prev, set()):
            violations.append(f"{k}: illegal verdict chain {prev} -> {verdict}")
        state[k] = verdict
        # bookkeeping for invariant 3
        if verdict == "winner":
            if arm in winner_mean:
                violations.append(f"{arm}: two winner verdicts in one arm")
            winner_mean[arm] = d["mean"]
        if verdict == "stopped_at_100":
            stopped_means.setdefault(arm, []).append(d["mean"])
    # 3. the winner's mean beats every stopped sibling's
    for arm, means in stopped_means.items():
        if arm not in winner_mean:
            violations.append(f"{arm}: stopped_at_100 verdicts exist but no winner was crowned")
        elif winner_mean[arm] < max(means) - 1e-9:
            violations.append(f"{arm}: winner mean {winner_mean[arm]:.4f} < a stopped sibling's "
                              f"{max(means):.4f}")
    # 4. no winner while a sibling is still undecided
    for arm in winner_mean:
        arm_keys = [bq.config_key(c) for c in bq.CONFIGS if c["arm"] == arm]
        undecided = [k for k in arm_keys if k not in state]
        if undecided:
            violations.append(f"{arm}: winner crowned while {len(undecided)} sibling "
                              f"configuration(s) undecided, e.g. {undecided[0]}")
    # 7. marker hygiene
    pend = pending_by_key_and_seed(sweep_id)
    for k, verdict in state.items():
        if verdict == "truncated" and pend.get(k):
            violations.append(f"{k}: truncated but {len(pend[k])} markers are still pending")
        if verdict == "stopped_at_100":
            tail = [s for s in pend.get(k, []) if s >= n_race]
            if tail:
                violations.append(f"{k}: stopped_at_100 but {len(tail)} markers with seed index "
                                  f">= {n_race} are still pending")
    known = {bq.config_key(c) for c in bq.CONFIGS}
    for k in state:
        if k not in known:
            violations.append(f"{k}: decided, but it is not one of this run's {len(known)} "
                              "configurations")
    return violations, state


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sweep_id", required=True)
    p.add_argument("--n_required", type=int, default=20)
    p.add_argument("--n_race", type=int, default=100)
    p.add_argument("--n_final", type=int, default=300)
    args = p.parse_args()
    violations, state = check(args.sweep_id, args.n_required, args.n_race, args.n_final)
    counts = {}
    for v in state.values():
        counts[v] = counts.get(v, 0) + 1
    print(f"{len(state)} decided configurations of {len(bq.CONFIGS)}: "
          + ", ".join(f"{v}={n}" for v, n in sorted(counts.items())) if state else
          f"0 decided configurations of {len(bq.CONFIGS)}")
    if violations:
        print(f"!!! {len(violations)} INVARIANT VIOLATION(S) !!!")
        for v in violations:
            print(f"  - {v}")
        sys.exit(1)
    print("all truncation invariants hold")


if __name__ == "__main__":
    main()
