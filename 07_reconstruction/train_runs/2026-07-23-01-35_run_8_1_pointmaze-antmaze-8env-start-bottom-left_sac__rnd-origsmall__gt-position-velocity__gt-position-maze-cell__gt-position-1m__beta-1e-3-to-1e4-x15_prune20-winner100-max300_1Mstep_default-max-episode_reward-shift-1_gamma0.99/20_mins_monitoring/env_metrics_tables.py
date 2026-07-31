#!/usr/bin/env python
"""Interim metrics tables for the point maze + ant maze train run 1 monitoring loop (the second of
the two CLI table kinds of the shared sweep-monitoring skill). Prints EIGHT tables, one per env
setup in build_queue.ENV_SETUPS_RUN1 order. Each table has one row per algorithm of that env,
showing the algorithm's CURRENT BEST config (best = highest mean per-episode extrinsic return over
its completed records), ranked by reward (best first). Repeatedly callable; keeps no state.

Metric columns — each "mean ± standard-error" over the best config's completed seeds, plus a final
completed-seeds N column. Standard error = sample std / sqrt(n) (0 when n < 2).
Definitions revised 2026-07-24 (user): the "current performance" columns use only the LAST 100
episodes of each run; ranking (and the racing prune score) stays the whole-run reward:
- reward           : score_of_record(d) = mean over ALL train_episode_history rows of
                     train/extrinsic_reward (the whole-run mean per-episode extrinsic return;
                     the RANKING column and the prune score)
- reward_last100   : per record = mean train/extrinsic_reward over the LAST 100 episodes
- success_rate     : per record = (# of the LAST 100 episodes with a truthy train/success) / 100
                     (denominator = however many of the last 100 exist when a run has fewer)
- steps_to_goal    : per record = mean train/steps_to_goal over its SUCCESSFUL rows (whole run);
                     records with zero successes are excluded from this column (its included-record
                     count is shown as "(n=..)" when it differs from N)
- maze_cell_cov%   : visit_counts/coverage_pct at the END of the run (the last eval snapshot; the
                     counts are cumulative, so this is the coverage after the final — i.e. last
                     100 — episodes; a last-100-only coverage is not in the record format)
- 1m_cov%          : visit_counts_1m/coverage_pct at the END of the run (same convention)

Row label spells the swept knob out: bonus algorithms -> "<algorithm> — bonus-weight-<beta>" (the
--beta intrinsic-bonus weight is the only swept knob); no_exploration -> "no_exploration — no swept
knob". A fully empty env table prints "no completed records yet"; algorithms of the env that have no
completed record yet are listed in a trailing note.

Usage:  python env_metrics_tables.py --sweep_id <id> [--snapshot]
  --snapshot also writes the text to 20_mins_monitoring/outputs/<YYYY-MM-DD-HH-MM>_env_metrics.txt

RUN_DIR_OVERRIDE (env var, testing only): when set, data/ and outputs/ are read/written under that
directory instead of the real run folder; build_queue and prune_controller are ALWAYS imported from
the real slurm/ (config source of truth). Used by the machinery_test harness.
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
sys.path.insert(0, os.path.join(REAL_RUN_DIR, "slurm"))
import build_queue        # noqa: E402  (CONFIGS / ENV_SETUPS_RUN1 / config_key — source of truth)
import prune_controller   # noqa: E402  (score_of_record / key_from_record — scoring + grouping)

DATA_RUN_DIR = os.environ.get("RUN_DIR_OVERRIDE", REAL_RUN_DIR)


def mean_se(vals):
    """(mean, standard error, n) of a list. SE = sample std / sqrt(n), 0 when n < 2; (None, None, 0)
    for an empty list. Sample std uses the (n-1) denominator (matches prune_controller.mean_std)."""
    n = len(vals)
    if n == 0:
        return None, None, 0
    m = sum(vals) / n
    if n < 2:
        return m, 0.0, n
    var = sum((v - m) ** 2 for v in vals) / (n - 1)
    return m, math.sqrt(var) / math.sqrt(n), n


def compact_record(d):
    """Reduce a full per-run record to a small dict carrying ONLY what the metrics tables need,
    so the loader never holds the ~1400-episode train_episode_history / eval_history of thousands of
    records in memory at once (that OOM-killed the 4G monitor at ~3000 records). All six metric
    helpers below read the precomputed "_"-prefixed scalars when present and fall back to computing
    from a full record otherwise (so the standalone CLI still works on raw JSON).

    before: {..., train_episode_history:[1400 dicts], eval_history:[20 dicts], + config fields}
    after : {env_setup, algorithm, beta, _score, _r100, _succ, _steps, _mcov, _cov1m}  (9 scalars)
    """
    return {
        "env_setup": d.get("env_setup"),
        "algorithm": d.get("algorithm"),
        "beta": d.get("beta"),
        "_score": prune_controller.score_of_record(d),     # whole-run mean per-episode return
        "_r100": record_reward_last100(d),                  # last-100 mean return
        "_succ": record_success_rate(d),                    # last-100 success fraction
        "_steps": record_steps_to_goal(d),                  # mean steps over successful episodes
        "_mcov": record_last_eval(d, "visit_counts/coverage_pct"),
        "_cov1m": record_last_eval(d, "visit_counts_1m/coverage_pct"),
    }


def load_completed(sweep_id):
    """{config_key: [compact_record, ...]} over completed per-run JSONs that have a computable score.
    Each record is reduced to scalars immediately (compact_record) so the loader's memory stays
    bounded by the record COUNT, not the total episode count. A missing "completed" field counts as
    complete (prune_controller convention); a completed=false checkpoint of a killed attempt is
    skipped, as is a record with no scorable episode."""
    local = os.path.join(DATA_RUN_DIR, "data", sweep_id, "local")
    by_key = {}
    for path in glob.glob(os.path.join(local, "*.json")):
        try:
            with open(path) as fh:
                d = json.load(fh)
        except (json.JSONDecodeError, OSError):
            continue
        if not d.get("completed", True):
            continue
        if prune_controller.score_of_record(d) is None:
            continue
        key = prune_controller.key_from_record(d)
        by_key.setdefault(key, []).append(compact_record(d))
        # d (with its big history lists) goes out of scope here and is freed before the next file
    return by_key


def score_of(d):
    """Whole-run mean per-episode return: the precomputed compact scalar, or compute from a full record."""
    return d["_score"] if "_score" in d else prune_controller.score_of_record(d)


def record_reward_last100(d):
    """Mean train/extrinsic_reward over the record's LAST 100 episodes (fewer if the run has fewer);
    None when the record has no episodes. Reads the compact scalar when present."""
    if "_r100" in d:
        return d["_r100"]
    rows = (d.get("train_episode_history") or [])[-100:]
    vals = [r["train/extrinsic_reward"] for r in rows if "train/extrinsic_reward" in r]
    if not vals:
        return None
    return sum(vals) / len(vals)


def record_success_rate(d):
    """Fraction of the record's LAST 100 episodes with a truthy train/success (denominator =
    however many of the last 100 exist); None when the record has no episodes. Compact-aware."""
    if "_succ" in d:
        return d["_succ"]
    rows = (d.get("train_episode_history") or [])[-100:]
    if not rows:
        return None
    return sum(1 for r in rows if r.get("train/success")) / len(rows)


def record_steps_to_goal(d):
    """Mean train/steps_to_goal over the record's SUCCESSFUL episodes; None when it had no success
    (so a zero-success record is excluded from the steps-to-goal column). Compact-aware."""
    if "_steps" in d:
        return d["_steps"]
    rows = d.get("train_episode_history") or []
    steps = [r["train/steps_to_goal"] for r in rows
             if r.get("train/success") and "train/steps_to_goal" in r]
    if not steps:
        return None
    return sum(steps) / len(steps)


def record_last_eval(d, field):
    """A field on the record's LAST eval_history row, or None if there is no eval row / no field.
    Compact-aware: the two coverage fields are precomputed onto compact records."""
    if field == "visit_counts/coverage_pct" and "_mcov" in d:
        return d["_mcov"]
    if field == "visit_counts_1m/coverage_pct" and "_cov1m" in d:
        return d["_cov1m"]
    evals = d.get("eval_history") or []
    if not evals:
        return None
    return evals[-1].get(field)


def env_algorithms(env_setup):
    """Ordered distinct algorithms of an env (build_queue.CONFIGS order)."""
    out, seen = [], set()
    for c in build_queue.CONFIGS:
        if c["env_setup"] == env_setup and c["algorithm"] not in seen:
            seen.add(c["algorithm"])
            out.append(c["algorithm"])
    return out


def algo_config_keys(env_setup, algorithm):
    """config_keys of (env_setup, algorithm) in CONFIGS order."""
    return [build_queue.config_key(c) for c in build_queue.CONFIGS
            if c["env_setup"] == env_setup and c["algorithm"] == algorithm]


def best_config(by_key, keys):
    """(config_key, [records]) of the key with the highest mean score among `keys`; (None, []) when
    none of the keys have any completed record."""
    best_k, best_recs, best_mean = None, [], None
    for k in keys:
        recs = by_key.get(k, [])
        if not recs:
            continue
        m = sum(score_of(r) for r in recs) / len(recs)
        if best_mean is None or m > best_mean:
            best_k, best_recs, best_mean = k, recs, m
    return best_k, best_recs


def row_label(algorithm, config_key):
    """Row label naming the current best config with the swept knob spelled out (see module doc)."""
    if algorithm == "no_exploration":
        return f"{algorithm} — no swept knob"
    beta = config_key.split("|")[2]                # env_setup|algorithm|beta
    return f"{algorithm} — bonus-weight-{beta}"


def cell_str(mean, se, prec):
    """A "mean ± se" string at the given precision, or an em dash for no data."""
    if mean is None:
        return "—"
    return f"{mean:.{prec}f} ± {se:.{prec}f}"


def build_row(algorithm, config_key, recs):
    """One metrics row (dict of formatted strings + the reward mean used to rank)."""
    # reward: every record in `recs` has a computable score by construction
    reward_vals = [score_of(r) for r in recs]
    r_mean, r_se, n = mean_se(reward_vals)
    # reward over the LAST 100 episodes of each record (end-of-training performance)
    r100_vals = [v for r in recs if (v := record_reward_last100(r)) is not None]
    r100_mean, r100_se, _ = mean_se(r100_vals)
    # success rate over the LAST 100 episodes of each record
    succ_vals = [v for r in recs if (v := record_success_rate(r)) is not None]
    s_mean, s_se, _ = mean_se(succ_vals)
    # steps-to-goal over records that had at least one success (others excluded)
    steps_vals = [v for r in recs if (v := record_steps_to_goal(r)) is not None]
    st_mean, st_se, st_n = mean_se(steps_vals)
    steps_txt = cell_str(st_mean, st_se, 1)
    if st_n and st_n != n:            # surface the sub-count when it differs from the row N
        steps_txt += f" (n={st_n})"
    # coverage from the last eval row of records that logged it
    mcov_vals = [v for r in recs if (v := record_last_eval(r, "visit_counts/coverage_pct")) is not None]
    mc_mean, mc_se, _ = mean_se(mcov_vals)
    cov1m_vals = [v for r in recs if (v := record_last_eval(r, "visit_counts_1m/coverage_pct")) is not None]
    c1_mean, c1_se, _ = mean_se(cov1m_vals)
    return {
        "label": row_label(algorithm, config_key),
        "reward": cell_str(r_mean, r_se, 2),
        "reward100": cell_str(r100_mean, r100_se, 2),
        "success": cell_str(s_mean, s_se, 3),
        "steps": steps_txt,
        "mcov": cell_str(mc_mean, mc_se, 2),
        "cov1m": cell_str(c1_mean, c1_se, 2),
        "N": str(n),
        "_reward_mean": r_mean,
    }


MCOLS = [("label", "algorithm — knob (best config)", "l"), ("reward", "reward", "r"),
         ("reward100", "reward_last100ep", "r"),
         ("success", "success_rate", "r"), ("steps", "steps_to_goal", "r"),
         ("mcov", "maze_cell_cov%", "r"), ("cov1m", "1m_cov%", "r"), ("N", "N", "r")]


def render_table(env_setup, rows, awaiting):
    """Format one env's ranked rows into an aligned plain-text table (string). `awaiting` lists the
    env's algorithms with no completed record yet."""
    lines = [f"## {env_setup}"]
    if not rows:
        lines.append("   (no completed records yet)")
        return "\n".join(lines)
    # width per column over header + every cell
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
    if awaiting:
        lines.append(f"   (awaiting first completed record: {', '.join(awaiting)})")
    return "\n".join(lines)


def build_all(sweep_id):
    """Full printable text: the eight per-env tables plus a metric legend footer."""
    by_key = load_completed(sweep_id)
    blocks = [f"# run 8.1 interim metrics  sweep_id={sweep_id}  {time.strftime('%Y-%m-%dT%H:%M:%S')}",
              "# reward = whole-run mean per-episode extrinsic return (RANKING column and prune score);",
              "# reward_last100ep = mean per-episode return over the LAST 100 episodes of each run;",
              "# success_rate = fraction of the LAST 100 episodes reaching the goal;",
              "# steps_to_goal = mean steps on SUCCESSFUL episodes, whole run (zero-success seeds "
              "excluded);",
              "# coverage = cumulative visit-count coverage % at the END of the run (after its final "
              "episodes; a last-100-only coverage is not in the record format).",
              ""]
    for env_setup in build_queue.ENV_SETUPS_RUN1:
        rows, awaiting = [], []
        for algorithm in env_algorithms(env_setup):
            keys = algo_config_keys(env_setup, algorithm)
            best_k, recs = best_config(by_key, keys)
            if best_k is None:
                awaiting.append(algorithm)
                continue
            rows.append(build_row(algorithm, best_k, recs))
        rows.sort(key=lambda r: r["_reward_mean"], reverse=True)   # rank by reward, best first
        blocks.append(render_table(env_setup, rows, awaiting))
        blocks.append("")
    return "\n".join(blocks)


def main():
    """Print the eight interim metrics tables; with --snapshot also save a timestamped copy."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep_id", required=True)
    ap.add_argument("--snapshot", action="store_true")
    args = ap.parse_args()
    text = build_all(args.sweep_id)
    print(text)
    if args.snapshot:
        out_dir = os.path.join(DATA_RUN_DIR, "20_mins_monitoring", "outputs")
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{time.strftime('%Y-%m-%d-%H-%M')}_env_metrics.txt")
        with open(path, "w") as fh:
            fh.write(text + "\n")
        print(f"[snapshot] {path}")


if __name__ == "__main__":
    main()
