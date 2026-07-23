#!/usr/bin/env python
"""Prune-correctness invariant checker (mandatory each monitoring tick — this run's racing
thresholds 20/100/300 differ from the sweep_prune skill defaults, so the controller is verified,
not trusted).

Re-verifies every decision line of slurm/prune_decisions_<sweep_id>.jsonl and the queue state:
1. a "pruned" decision has n >= n_floor, recomputes mean + 2.576*std/sqrt(n) < bar_mean from its
   own logged numbers, and never targets the cell's bar (best) config;
2. a "winner_only" decision has n >= n_winner for the cut config AND bar_n >= n_winner for the
   winner it lost to;
3. every decided config belongs to a multi-config cell (a single-config SAC cell can never prune)
   and its config_key's first two fields equal the logged cell;
4. no decided config still has pending queue entries (the controller must have moved them all).

Prints one loud VIOLATION line per failed invariant and exits 1; prints a PASS summary otherwise.

Usage:  python prune_check.py --sweep_id <id> [--n_floor 20] [--n_winner 100]
"""
import argparse
import glob
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RUN_DIR = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(RUN_DIR, "slurm"))
import build_queue  # noqa: E402

Z99 = 2.576


def check_decisions(decision_lines, n_floor, n_winner, cell_sizes):
    """Pure checker: return the list of violation strings for the parsed decision lines.
    cell_sizes maps "env_setup|algorithm" -> number of configs in that cell."""
    violations = []
    for i, d in enumerate(decision_lines):
        tag = f"line {i + 1} ({d.get('config_key')})"
        verdict = d.get("verdict")
        if verdict not in ("pruned", "winner_only"):
            continue
        key = d.get("config_key", "")
        cell = d.get("cell", "")
        # invariant 3a: the key's cell fields equal the logged cell
        if "|".join(key.split("|")[:2]) != cell:
            violations.append(f"VIOLATION {tag}: config_key does not belong to logged cell {cell!r}")
        # invariant 3b: never a single-config cell
        if cell_sizes.get(cell, 0) < 2:
            violations.append(f"VIOLATION {tag}: decision in a single-config cell {cell!r}")
        # invariant 1/2: thresholds and the recomputed trigger
        if verdict == "pruned":
            if d["n"] < n_floor:
                violations.append(f"VIOLATION {tag}: pruned with n={d['n']} < floor {n_floor}")
            upper = d["mean"] + Z99 * d["std"] / math.sqrt(d["n"])
            if not upper < d["bar_mean"]:
                violations.append(
                    f"VIOLATION {tag}: recomputed upper {upper:.4f} is NOT below bar {d['bar_mean']:.4f}")
            if abs(upper - d.get("upper_99", upper)) > 1e-6:
                violations.append(f"VIOLATION {tag}: logged upper_99 does not match its own mean/std/n")
            if d.get("bar_n", 0) < n_floor:
                violations.append(f"VIOLATION {tag}: bar config had n={d.get('bar_n')} < floor {n_floor}")
            if key == d.get("bar_key"):
                violations.append(f"VIOLATION {tag}: the cell's best config was pruned")
        if verdict == "winner_only":
            if d["n"] < n_winner:
                violations.append(f"VIOLATION {tag}: winner_only cut with n={d['n']} < {n_winner}")
            if d.get("bar_n", 0) < n_winner:
                violations.append(f"VIOLATION {tag}: winner had n={d.get('bar_n')} < {n_winner}")
            if key == d.get("bar_key"):
                violations.append(f"VIOLATION {tag}: the winner cut itself")
    return violations


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sweep_id", required=True)
    p.add_argument("--n_floor", type=int, default=20)
    p.add_argument("--n_winner", type=int, default=100)
    args = p.parse_args()
    log_path = os.path.join(RUN_DIR, "slurm", f"prune_decisions_{args.sweep_id}.jsonl")
    lines = []
    if os.path.exists(log_path):
        with open(log_path) as fh:
            for raw in fh:
                try:
                    lines.append(json.loads(raw))
                except json.JSONDecodeError:
                    print(f"VIOLATION: unparseable decision line: {raw[:120]!r}")
    cell_sizes = {}
    for c in build_queue.CONFIGS:
        cell = f"{c['env_setup']}|{c['algorithm']}"
        cell_sizes[cell] = cell_sizes.get(cell, 0) + 1
    violations = check_decisions(lines, args.n_floor, args.n_winner, cell_sizes)
    # invariant 4: no decided config keeps pending entries
    pending = os.path.join(RUN_DIR, "queue", args.sweep_id, "pending")
    decided_keys = {d["config_key"] for d in lines if d.get("verdict") in ("pruned", "winner_only")}
    for key in sorted(decided_keys):
        spec = next(c for c in build_queue.CONFIGS if build_queue.config_key(c) == key)
        left = glob.glob(os.path.join(pending, f"*_of_*_{build_queue.label(spec)}_seed*.json"))
        if left:
            violations.append(f"VIOLATION: decided config {key} still has {len(left)} pending entries")
    for v in violations:
        print(v)
    n_dec = len(decided_keys)
    if violations:
        print(f"prune_check: {len(violations)} VIOLATION(S) across {n_dec} decided configs")
        sys.exit(1)
    print(f"prune_check PASS: {n_dec} decided configs, all invariants hold "
          f"(floor {args.n_floor}, winner-only {args.n_winner})")


if __name__ == "__main__":
    main()
