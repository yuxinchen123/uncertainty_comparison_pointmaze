#!/usr/bin/env python
"""Stage-1 decision invariant checker (mandatory each monitoring tick — this run replaces the
sweep_prune skill's cell-relative bar with a FROZEN external bar, so the controller is verified,
not trusted).

Re-verifies every decision line of slurm/stage1_decisions_<sweep_id>.jsonl and the queue state:
1. a "pruned" decision has n >= n_required, its logged upper_99 equals mean + 2.576*std/sqrt(n),
   the recomputed upper is BELOW the logged bar, and the logged bar equals FROZEN_BARS.json's mean
   for the env (and the logged bars_sha256 equals the current file's hash — a mid-run bar change
   is a violation);
2. a "survivor" decision has n >= n_target and its recomputed upper is NOT below the bar;
3. every decided config_key is a member of build_queue.CONFIGS (never the exempt baseline arm) and
   at most one decision exists per config;
4. no pruned config still has pending_1m entries (the controller must have moved them all);
5. a survivor has no pending_1m entries either (its 100 seeds were all claimed).

Prints one loud VIOLATION line per failed invariant and exits 1; prints a PASS summary otherwise.

Usage:  python stage1_check.py --sweep_id <id> [--n_required 30] [--n_target 100]
"""
import argparse
import glob
import hashlib
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RUN_DIR = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(RUN_DIR, "slurm"))
import build_queue  # noqa: E402

Z99 = 2.576


def check_decisions(decision_lines, n_required, n_target, bars, bars_sha, valid_keys):
    """Pure checker: return the list of violation strings for the parsed decision lines.
    bars maps env_setup -> frozen mean; valid_keys is the set of task-S config_keys."""
    violations = []
    seen = {}
    for i, d in enumerate(decision_lines):
        tag = f"line {i + 1} ({d.get('config_key')})"
        verdict = d.get("verdict")
        if verdict not in ("pruned", "survivor"):
            continue
        key = d.get("config_key", "")
        env = d.get("env_setup", "")
        # invariant 3: known task-S config (the baseline arm can never be decided), one decision each
        if key not in valid_keys:
            violations.append(f"VIOLATION {tag}: config_key is not a task-S sweep config")
        if key in seen:
            violations.append(f"VIOLATION {tag}: second decision for {key} (first: {seen[key]})")
        seen[key] = verdict
        if key.split("|")[0] != env:
            violations.append(f"VIOLATION {tag}: env_setup field does not match the config_key")
        # invariant 1a: the logged bar equals the frozen file's value and hash
        if env in bars and abs(d.get("bar", float("nan")) - bars[env]) > 1e-9:
            violations.append(f"VIOLATION {tag}: logged bar {d.get('bar')} != frozen {bars[env]}")
        if d.get("bars_sha256") != bars_sha:
            violations.append(f"VIOLATION {tag}: bars_sha256 does not match the current FROZEN_BARS.json")
        # invariant 1b/2: thresholds and the recomputed trigger
        upper = d["mean"] + Z99 * d["std"] / math.sqrt(d["n"])
        if abs(upper - d.get("upper_99", upper)) > 1e-6:
            violations.append(f"VIOLATION {tag}: logged upper_99 does not match its own mean/std/n")
        if verdict == "pruned":
            if d["n"] < n_required:
                violations.append(f"VIOLATION {tag}: pruned with n={d['n']} < required {n_required}")
            if not upper < d["bar"]:
                violations.append(
                    f"VIOLATION {tag}: recomputed upper {upper:.4f} is NOT below bar {d['bar']:.4f}")
        if verdict == "survivor":
            if d["n"] < n_target:
                violations.append(f"VIOLATION {tag}: survivor with n={d['n']} < target {n_target}")
            if upper < d["bar"]:
                violations.append(
                    f"VIOLATION {tag}: survivor whose recomputed upper {upper:.4f} is below the bar "
                    f"{d['bar']:.4f} (should have been pruned)")
    return violations


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sweep_id", required=True)
    p.add_argument("--n_required", type=int, default=30)
    p.add_argument("--n_target", type=int, default=100)
    args = p.parse_args()
    # the frozen bars (hash of the exact bytes, matching stage1_controller.bars())
    bars_path = os.path.join(RUN_DIR, "slurm", "FROZEN_BARS.json")
    if not os.path.exists(bars_path):
        print(f"VIOLATION: {bars_path} missing while decisions may exist")
        sys.exit(1)
    raw = open(bars_path, "rb").read()
    bars = {env: v["mean"] for env, v in json.loads(raw)["bars"].items()}
    bars_sha = hashlib.sha256(raw).hexdigest()
    log_path = os.path.join(RUN_DIR, "slurm", f"stage1_decisions_{args.sweep_id}.jsonl")
    lines = []
    if os.path.exists(log_path):
        with open(log_path) as fh:
            for raw_line in fh:
                try:
                    lines.append(json.loads(raw_line))
                except json.JSONDecodeError:
                    print(f"VIOLATION: unparseable decision line: {raw_line[:120]!r}")
    valid_keys = {build_queue.config_key(c) for c in build_queue.CONFIGS}
    violations = check_decisions(lines, args.n_required, args.n_target, bars, bars_sha, valid_keys)
    # invariants 4/5: no decided config keeps pending_1m entries
    pending = os.path.join(RUN_DIR, "queue", args.sweep_id, "pending_1m")
    decided_keys = {d["config_key"] for d in lines if d.get("verdict") in ("pruned", "survivor")}
    for key in sorted(decided_keys & valid_keys):
        spec = next(c for c in build_queue.CONFIGS if build_queue.config_key(c) == key)
        left = glob.glob(os.path.join(pending, f"*_of_*_{build_queue.label(spec)}_seed*.json"))
        if left:
            violations.append(f"VIOLATION: decided config {key} still has {len(left)} pending entries")
    for v in violations:
        print(v)
    n_dec = len(decided_keys)
    if violations:
        print(f"stage1_check: {len(violations)} VIOLATION(S) across {n_dec} decided configs")
        sys.exit(1)
    print(f"stage1_check PASS: {n_dec} decided configs, all invariants hold "
          f"(required {args.n_required}, target {args.n_target})")


if __name__ == "__main__":
    main()
