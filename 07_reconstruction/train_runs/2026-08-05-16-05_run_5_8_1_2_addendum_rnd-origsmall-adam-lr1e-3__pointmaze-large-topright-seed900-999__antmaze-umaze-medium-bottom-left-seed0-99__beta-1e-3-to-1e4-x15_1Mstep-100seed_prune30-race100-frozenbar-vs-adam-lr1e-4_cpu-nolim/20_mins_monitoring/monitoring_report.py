#!/usr/bin/env python
"""The per-tick monitoring report for this run — the two tables of the shared sweep-monitoring skill.

Table 1 (status): one row per environment — configurations, verdicts so far, completed runs, and how
much of the queue is left.
Table 2 (metrics): one block per environment — every configuration ranked by its mean score under
that environment's own score rule, with the standard error, the completed-seed count, the one-sided
99% upper bound the truncation rule uses, and the verdict so far. The block header restates the
frozen bar and how many of the environment's 15 configurations currently sit at or above it.

The writeup's tables for this run are generated from these same functions, so the document and the
monitoring report can never disagree.

Marking follows the analysis convention: within each environment block, best mean score in bold and
second best underlined; the seed-count and verdict columns are never marked.

Usage:
  python monitoring_report.py --sweep_id <id>              # print to stdout
  python monitoring_report.py --sweep_id <id> --snapshot   # also write outputs/<stamp>_monitoring.md
"""
import argparse
import glob
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RUN_DIR = os.path.dirname(HERE)
SLURM = os.path.join(RUN_DIR, "slurm")
sys.path.insert(0, SLURM)
import build_queue as bq            # noqa: E402
import score_rules                  # noqa: E402
import truncation_controller as tc  # noqa: E402

ENV_SHORT = {
    "initial_single_large_pointmaze_max_400": "PointMaze Large (top-right, train run 5)",
    "AntMaze_UMaze-v5_start_bottom_left": "AntMaze UMaze (bottom-left, train run 1.2)",
    "AntMaze_Medium-v5_start_bottom_left": "AntMaze Medium (bottom-left, train run 1.2)",
}


def load_all(sweep_id):
    """{config_key: [score, ...]} over the sweep's completed records (the controller's own loader)."""
    return tc.load_scores(sweep_id)


def queue_counts(sweep_id):
    """{state: count} over the sweep's queue directories."""
    q = os.path.join(RUN_DIR, "queue", sweep_id)
    out = {}
    for state in ("pending", "running", "done", "failed", "pruned"):
        path = os.path.join(q, state)
        out[state] = len(os.listdir(path)) if os.path.isdir(path) else 0
    return out


def verdicts(sweep_id):
    """{config_key: verdict} from the decision log."""
    path = os.path.join(SLURM, f"truncation_decisions_{sweep_id}.jsonl")
    out = {}
    if os.path.exists(path):
        with open(path) as fh:
            for line in fh:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if d.get("verdict") in ("truncated", "survivor"):
                    out[d["config_key"]] = d["verdict"]
    return out


def bars():
    """{env: (mean, bonus weight, score rule)} from the frozen bars file."""
    doc = json.load(open(os.path.join(SLURM, "FROZEN_BARS.json")))
    return {env: (v["mean"], v["beta"], v["score_rule"]) for env, v in doc["bars"].items()}


def stats(vals):
    """(mean, standard error, n, one-sided 99% upper bound) of a score list; zeros for an empty one."""
    n = len(vals)
    if n == 0:
        return 0.0, 0.0, 0, 0.0
    mean, sd = tc.mean_std(vals)
    se = sd / math.sqrt(n)
    return mean, se, n, mean + tc.Z99 * se


def status_rows(sweep_id):
    """One row per environment: configurations, verdicts, completed runs."""
    scores, decided = load_all(sweep_id), verdicts(sweep_id)
    rows = []
    for env in bq.ENV_SETUPS:
        keys = [bq.config_key(c) for c in bq.CONFIGS if c["env_setup"] == env]
        rows.append({
            "environment": ENV_SHORT[env],
            "configurations": len(keys),
            "undecided": sum(1 for k in keys if k not in decided),
            "truncated": sum(1 for k in keys if decided.get(k) == "truncated"),
            "survivors": sum(1 for k in keys if decided.get(k) == "survivor"),
            "completed runs": sum(len(scores.get(k, [])) for k in keys),
        })
    return rows


def metric_rows(sweep_id, env):
    """Every configuration of one environment, ranked by mean score (best first)."""
    scores, decided = load_all(sweep_id), verdicts(sweep_id)
    rows = []
    for cfg in [c for c in bq.CONFIGS if c["env_setup"] == env]:
        k = bq.config_key(cfg)
        mean, se, n, upper = stats(scores.get(k, []))
        rows.append({"bonus weight": cfg["beta"], "mean": mean, "se": se, "n": n,
                     "upper 99%": upper, "verdict": decided.get(k, "undecided" if n else "no data")})
    return sorted(rows, key=lambda r: (-r["n"] > 0, -r["mean"]))


def render_table(rows, columns):
    """A Markdown table from a list of dicts and a column order."""
    head = "| " + " | ".join(columns) + " |"
    rule = "|" + "|".join("---" for _ in columns) + "|"
    body = ["| " + " | ".join(str(r[c]) for c in columns) + " |" for r in rows]
    return "\n".join([head, rule] + body)


def fmt(value, best, second):
    """Bold the best mean, underline the second best (the analysis-convention marking)."""
    text = f"{value:.4f}"
    if value == best:
        return f"**{text}**"
    if value == second:
        return f"<u>{text}</u>"
    return text


def report(sweep_id):
    """The whole report as one Markdown string."""
    counts, BARS = queue_counts(sweep_id), bars()
    lines = [f"# Monitoring report — sweep {sweep_id}", "",
             f"Generated {time.strftime('%Y-%m-%dT%H:%M:%S')}.", "",
             "## Queue", "",
             render_table([counts], ["pending", "running", "done", "failed", "pruned"]), "",
             "## Status by environment", "",
             render_table(status_rows(sweep_id),
                          ["environment", "configurations", "undecided", "truncated", "survivors",
                           "completed runs"]), ""]
    lines += ["## Scores by environment", ""]
    for env in bq.ENV_SETUPS:
        bar, bar_beta, rule = BARS[env]
        rows = metric_rows(sweep_id, env)
        scored = sorted({r["mean"] for r in rows if r["n"] > 0}, reverse=True)
        best = scored[0] if scored else None
        second = scored[1] if len(scored) > 1 else None
        at_or_above = sum(1 for r in rows if r["n"] > 0 and r["mean"] >= bar)
        lines += [f"### {ENV_SHORT[env]}", "",
                  f"Score rule: {score_rules.RULE_DESCRIPTION[rule]}.", "",
                  f"Frozen bar {bar:.4f} — the Adam 1e-4 winner at bonus weight {bar_beta}. "
                  f"{at_or_above} of {len(rows)} configurations sit at or above it so far.", ""]
        table = [{"bonus weight": r["bonus weight"],
                  "mean score": fmt(r["mean"], best, second) if r["n"] else "—",
                  "standard error": f"{r['se']:.4f}" if r["n"] else "—",
                  "completed seeds": r["n"],
                  "99% upper bound": f"{r['upper 99%']:.4f}" if r["n"] else "—",
                  "verdict": r["verdict"]} for r in rows]
        lines += [render_table(table, ["bonus weight", "mean score", "standard error",
                                       "completed seeds", "99% upper bound", "verdict"]), ""]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sweep_id", required=True)
    p.add_argument("--snapshot", action="store_true",
                   help="also write outputs/<timestamp>_monitoring.md")
    args = p.parse_args()
    text = report(args.sweep_id)
    print(text)
    if args.snapshot:
        out = os.path.join(HERE, "outputs",
                           f"{time.strftime('%Y-%m-%d-%H-%M')}_monitoring.md")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as fh:
            fh.write(text + "\n")
        print(f"\n[snapshot] {out}")


if __name__ == "__main__":
    main()
