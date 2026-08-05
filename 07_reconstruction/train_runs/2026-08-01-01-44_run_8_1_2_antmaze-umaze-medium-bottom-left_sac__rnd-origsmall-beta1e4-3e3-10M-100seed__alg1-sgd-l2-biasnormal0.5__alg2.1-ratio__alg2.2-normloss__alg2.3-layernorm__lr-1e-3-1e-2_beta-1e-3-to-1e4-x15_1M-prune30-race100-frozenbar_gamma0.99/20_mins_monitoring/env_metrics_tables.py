#!/usr/bin/env python
"""Interim metrics tables for the train run 8.1.2 monitoring loop (the second of the two CLI table
kinds of the shared sweep-monitoring skill). Prints TWO tables, one per env setup in
build_queue.ENV_SETUPS_RUN12 order. Each table has one row per ARM — the task-R baseline plus the
four task-S arms (alg1, alg2.1, alg2.2, alg2.3) — showing that arm's CURRENT BEST config, best =
highest mean whole-run return over the config's completed records. An arm with no completed record
yet has no row and is named in a trailing note. Repeatedly callable; keeps no state.

Columns:
- whole-run reward : "mean ± standard error" of score_of_record(d) = the mean per-episode
                     train/extrinsic_reward over ALL train_episode_history rows (the stage-1 racing
                     score, so this column and the controller rank configs identically). Standard
                     error = sample standard deviation / sqrt(n), 0 when n < 2.
- completed seeds  : n, the number of completed records behind the row (bookkeeping, never marked)
- verdict so far   : the stage-1 verdict of that config — undecided / pruned / survivor; "N/A" on
                     the baseline row, which is exempt from every decision

Row label spells the config's knobs out (arm, learning rate, bonus weight), e.g.
"alg2.2 — lr 0.01, bonus-weight 3". On the baseline row the learning rate is its adam rate (the
task-S arms use plain SGD) and its runs are 10M steps long — both stated in the legend.

Under each table one line reports the env's frozen bar and how many of its task-S configs currently
sit at or above it:
  "frozen bar: -689.7404 (beta 10000); configs above bar so far: 7 of 120"
The count is over configs with at least one completed record whose mean is >= the bar; the
denominator is the env's task-S configs, i.e. decided (pruned + survivor) + undecided.

CAVEAT carried in the printed legend: the baseline row's runs are 10M-step task-R runs while every
task-S row is a 1M-step run, and the frozen bar is the run-8.1 RND winner's 1M value — so the
baseline row is context, not a like-for-like comparison against the arms in the same table.

Usage:  python env_metrics_tables.py --sweep_id <id> [--snapshot]
  --snapshot also writes the text to 20_mins_monitoring/outputs/<YYYY-MM-DD-HH-MM>_env_metrics.txt

RUN_DIR_OVERRIDE (env var, testing only): when set, data/, slurm/ (the decisions log and
FROZEN_BARS.json) and outputs/ are read/written under that directory instead of the real run folder;
build_queue and stage1_controller are ALWAYS imported from the real slurm/ (source of truth).
"""
import argparse
import glob
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REAL_RUN_DIR = os.path.dirname(HERE)
sys.path.insert(0, HERE)                                    # status_table (the decisions reader)
sys.path.insert(0, os.path.join(REAL_RUN_DIR, "slurm"))     # build_queue / stage1_controller
import build_queue        # noqa: E402  (CONFIGS / BASELINE_CONFIGS / ARMS / config_key — the source)
import stage1_controller  # noqa: E402  (score_of_record — the racing/ranking score)
import status_table       # noqa: E402  (decision_verdicts / run_dir — one decisions reader)

NA = "N/A"
# the arms of every env in table order: the task-R baseline first, then the four task-S arms
ARM_ORDER = ["baseline"] + list(build_queue.ARMS)


def run_dir():
    """The run folder whose data/, slurm/ and outputs/ these tables read (RUN_DIR_OVERRIDE redirects
    it for tests). Same resolution as status_table.run_dir."""
    return status_table.run_dir()


def mean_se(vals):
    """(mean, standard error, n) of a list. Standard error = sample standard deviation / sqrt(n),
    0 when n < 2; (None, None, 0) for an empty list. The sample standard deviation uses the (n-1)
    denominator, matching stage1_controller.mean_std."""
    n = len(vals)
    if n == 0:
        return None, None, 0
    m = sum(vals) / n
    if n < 2:
        return m, 0.0, n
    var = sum((v - m) ** 2 for v in vals) / (n - 1)
    return m, math.sqrt(var) / math.sqrt(n), n


def final_score_of_record(d):
    """The record's FINAL reward: mean per-episode extrinsic return over the last min(100, E)
    training episodes (the run-1.1 tables' last-100 metric), or None with no scorable episode."""
    rows = d.get("train_episode_history") or []
    vals = [r["train/extrinsic_reward"] for r in rows if "train/extrinsic_reward" in r]
    if not vals:
        return None
    return sum(vals[-100:]) / len(vals[-100:])


def compact_record(d):
    """Reduce a full per-run record to the five scalars these tables need, so the loader never holds
    the long per-episode history of thousands of records at once (a task-R record carries ~10x the
    episodes of a task-S record, so this is what keeps the monitor job's memory bounded by the record
    COUNT rather than the total episode count).

    before: {"env_setup":..., "beta":"3", "rnd_lr":"0.01", "rnd_optimizer":"sgd", "completed":true,
             "total_timesteps":1000000, "train_episode_history":[~1400 dicts], "eval_history":[20 dicts]}
    after : {"key":"AntMaze_UMaze-v5_start_bottom_left|alg2.2|lr0.01|b3", "score":-688.4,
             "score100":-686.1, "completed":True, "total_timesteps":1000000}
    """
    return {
        "key": build_queue.key_from_record(d),
        "score": stage1_controller.score_of_record(d),
        "score100": final_score_of_record(d),
        "completed": d.get("completed", True),
        "total_timesteps": int(d.get("total_timesteps", 0)),
    }


def load_completed(sweep_id):
    """{config_key: [compact_record, ...]} over completed per-run JSONs that have a computable score.
    Each record is reduced to scalars immediately (compact_record) and the full record is dropped. A
    missing "completed" field counts as complete (the logging convention); a completed=false
    checkpoint of a killed attempt is skipped, as is a record with no scorable episode."""
    local = os.path.join(run_dir(), "data", sweep_id, "local")
    by_key = {}
    for path in glob.glob(os.path.join(local, "*.json")):
        try:
            with open(path) as fh:
                d = json.load(fh)
        except (json.JSONDecodeError, OSError):
            continue  # a record mid-flush this cycle; picked up next tick
        if not d.get("completed", True):
            continue
        if stage1_controller.score_of_record(d) is None:
            continue
        rec = compact_record(d)
        by_key.setdefault(rec["key"], []).append(rec)
        # d (with its big history lists) goes out of scope here and is freed before the next file
    return by_key


def arm_config_keys(env_setup, arm):
    """config_keys of (env_setup, arm) in build_queue order — from BASELINE_CONFIGS for the baseline
    arm, from CONFIGS for a task-S arm."""
    specs = build_queue.BASELINE_CONFIGS if arm == "baseline" else build_queue.CONFIGS
    return [build_queue.config_key(c) for c in specs
            if c["env_setup"] == env_setup and c["arm"] == arm]


def best_config(by_key, keys):
    """(config_key, [records]) of the key with the highest mean score among `keys`; (None, []) when
    none of the keys has a completed record yet."""
    best_k, best_recs, best_mean = None, [], None
    for k in keys:
        recs = by_key.get(k, [])
        if not recs:
            continue
        m = sum(r["score"] for r in recs) / len(recs)
        if best_mean is None or m > best_mean:
            best_k, best_recs, best_mean = k, recs, m
    return best_k, best_recs


def row_label(arm, config_key):
    """Row label naming the best config with its knobs spelled out. config_key is
    env_setup|arm|lr<lr>|b<beta>, so fields 2 and 3 carry the learning rate and the bonus weight (the
    baseline arm's learning rate is its adam rate; the legend carries that and its 10M length).
      before: ("alg2.2", "AntMaze_UMaze-v5_start_bottom_left|alg2.2|lr0.01|b3")
      after : "alg2.2 — lr 0.01, bonus-weight 3"
      before: ("baseline", "AntMaze_UMaze-v5_start_bottom_left|baseline|lr0.0001|b10000")
      after : "baseline — lr 0.0001, bonus-weight 10000"
    """
    _, _, lr_field, beta_field = config_key.split("|")
    lr, beta = lr_field[len("lr"):], beta_field[len("b"):]
    return f"{arm} — lr {lr}, bonus-weight {beta}"


def cell_str(mean, se, prec):
    """A "mean ± standard error" string at the given precision, or an em dash for no data."""
    if mean is None:
        return "—"
    return f"{mean:.{prec}f} ± {se:.{prec}f}"


def build_row(arm, config_key, recs, verdict):
    """One metrics row: the formatted whole-run and final (last-100) reward cells, the completed-seed
    count, the stage-1 verdict, plus the raw means (kept under "_raw" so the report can rank/mark
    both reward columns)."""
    r_mean, r_se, n = mean_se([r["score"] for r in recs])
    r100_mean, r100_se, _ = mean_se([r["score100"] for r in recs if r["score100"] is not None])
    return {
        "label": row_label(arm, config_key),
        "reward": cell_str(r_mean, r_se, 2),
        "reward100": cell_str(r100_mean, r100_se, 2),
        "N": str(n),
        "verdict": verdict,
        "_reward_mean": r_mean,
        "_raw": {"reward": r_mean, "reward100": r100_mean},
    }


def frozen_bars():
    """{env_setup: (bar mean, winning bonus weight)} from slurm/FROZEN_BARS.json — the same committed
    file the stage-1 controller decides against. Hard-fails when it is missing: a monitoring report
    must never invent a bar."""
    path = os.path.join(run_dir(), "slurm", "FROZEN_BARS.json")
    if not os.path.exists(path):
        sys.exit(f"missing {path} — the frozen bars are required to report against them")
    with open(path) as fh:
        doc = json.load(fh)
    return {env: (v["mean"], v["beta"]) for env, v in doc["bars"].items()}


def above_bar_counts(env_setup, by_key, bar):
    """(k, total) for the env's bar line: k = the env's task-S configs whose current mean whole-run
    return is at or above the frozen bar (a config with no completed record is not counted); total =
    the env's task-S configs, i.e. decided (pruned + survivor) plus undecided."""
    keys = [build_queue.config_key(c) for c in build_queue.CONFIGS
            if c["env_setup"] == env_setup]
    k = 0
    for key in keys:
        recs = by_key.get(key, [])
        if recs and sum(r["score"] for r in recs) / len(recs) >= bar:
            k += 1
    return k, len(keys)


def bar_line(env_setup, by_key):
    """The one-line frozen-bar summary printed under an env's table."""
    bar, beta = frozen_bars()[env_setup]
    k, total = above_bar_counts(env_setup, by_key, bar)
    return (f"frozen bar: {bar:.4f} (beta {beta}); configs above bar so far: {k} of {total}")


def env_rows(env_setup, by_key, verdicts):
    """(rows, awaiting) for one env: one row per arm's current best config, ranked by whole-run reward
    (best first); `awaiting` lists the arms with no completed record yet."""
    rows, awaiting = [], []
    for arm in ARM_ORDER:
        best_k, recs = best_config(by_key, arm_config_keys(env_setup, arm))
        if best_k is None:
            awaiting.append(arm)
            continue
        # the baseline arm is exempt from stage-1 decisions -> N/A, never a count and never a dash
        verdict = NA if arm == "baseline" else verdicts.get(best_k, "undecided")
        rows.append(build_row(arm, best_k, recs, verdict))
    rows.sort(key=lambda r: r["_reward_mean"], reverse=True)
    return rows, awaiting


MCOLS = [("label", "arm — knobs (best config)", "l"), ("reward", "whole-run reward", "r"),
         ("reward100", "final reward", "r"),
         ("N", "completed seeds", "r"), ("verdict", "verdict so far", "l")]


def render_table(env_setup, rows, awaiting, bar_text):
    """Format one env's ranked rows into an aligned plain-text table (string), with the frozen-bar
    line underneath. `awaiting` lists the arms with no completed record yet."""
    lines = [f"## {env_setup}"]
    if not rows:
        lines.append("   (no completed records yet)")
        lines.append(f"   {bar_text}")
        return "\n".join(lines)
    # width per column over the header and every cell
    widths = {}
    for key, header, _ in MCOLS:
        widths[key] = max([len(header)] + [len(str(r[key])) for r in rows])

    def fmt_row(r):
        parts = []
        for key, _, align in MCOLS:
            s = str(r[key])
            parts.append(s.ljust(widths[key]) if align == "l" else s.rjust(widths[key]))
        return "  ".join(parts)

    lines.append(fmt_row({k: h for k, h, _ in MCOLS}))
    lines.append("  ".join("-" * widths[k] for k, _, _ in MCOLS))
    lines += [fmt_row(r) for r in rows]
    lines.append(f"   {bar_text}")
    if awaiting:
        lines.append(f"   (awaiting first completed record: {', '.join(awaiting)})")
    return "\n".join(lines)


def build_all(sweep_id):
    """Full printable text: the two per-env tables plus the metric legend footer."""
    by_key = load_completed(sweep_id)
    verdicts = status_table.decision_verdicts(sweep_id)
    blocks = [f"# run 8.1.2 interim metrics  sweep_id={sweep_id}  "
              f"{time.strftime('%Y-%m-%dT%H:%M:%S')}",
              "# whole-run reward = mean per-episode extrinsic return over ALL training episodes of a",
              "#   run (the stage-1 racing score); cells are mean ± standard error over completed seeds;",
              "# verdict so far = the stage-1 verdict of that config (undecided / pruned / survivor);",
              "#   the baseline arm is exempt from decisions (N/A);",
              "# the baseline rows are 10M-step task-R runs and the arm rows are 1M-step task-S runs,",
              "#   so the baseline row is context, not a like-for-like comparison.",
              ""]
    for env_setup in build_queue.ENV_SETUPS_RUN12:
        rows, awaiting = env_rows(env_setup, by_key, verdicts)
        blocks.append(render_table(env_setup, rows, awaiting, bar_line(env_setup, by_key)))
        blocks.append("")
    return "\n".join(blocks)


def main():
    """Print the two interim metrics tables; with --snapshot also save a timestamped copy."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep_id", required=True)
    ap.add_argument("--snapshot", action="store_true")
    args = ap.parse_args()
    text = build_all(args.sweep_id)
    print(text)
    if args.snapshot:
        out_dir = os.path.join(run_dir(), "20_mins_monitoring", "outputs")
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{time.strftime('%Y-%m-%d-%H-%M')}_env_metrics.txt")
        with open(path, "w") as fh:
            fh.write(text + "\n")
        print(f"[snapshot] {path}")


if __name__ == "__main__":
    main()
