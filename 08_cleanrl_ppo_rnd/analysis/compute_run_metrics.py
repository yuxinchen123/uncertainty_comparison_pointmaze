"""Compute the end-of-run summaries from the records, after the runs have finished.

The trainer deliberately does not decide what "the performance of a run" is. It stores enough for
any reasonable definition to be computed exactly afterwards, and this script computes them.

The trainer logs exactly one performance number per row: `train/mean_extrinsic_reward`, a trailing
mean over the last 200 finished episodes. Everything below is derived from that series here, at the
aggregation and selection stage, so the definition of "the result of a run" is a decision made in
analysis rather than baked into the training code.

THE TWO CANDIDATE FINAL METRICS FOR A RUN, both derived from that one logged series:

1. **Final performance** — the LAST row's value. The policy at the end of the run.
2. **Whole-run average performance** — the mean of every row's value. The run's typical performance
   across its whole trajectory.

Neither is privileged. They answer different questions and this script reports both side by side:

| metric | what it answers | why it can mislead on its own |
|---|---|---|
| final performance | how good is the policy at the end | one row, and Montezuma's return is high-variance across episodes |
| whole-run average | how the run performed throughout | punishes a slow starter that ends strong, because early zeros stay in the average |
| best row | did the run ever find the good behaviour at all | rewards a single lucky interval |

One property of the whole-run average worth knowing when reading it: it weights every logged row
equally, so it is an average over TIME rather than over episodes. Rows late in a run cover the same
wall-clock interval as early ones but not necessarily the same number of episodes, so a run whose
episodes get longer is weighted slightly differently than a per-episode average would weight it.

Aggregation across seeds uses the median as well as the mean, because a sparse-reward Atari task
produces a long-tailed distribution across seeds where one lucky seed moves the mean a long way.

Run:
    PYTHONNOUSERSITE=1 /p/rlprojects/RND/.venvs/exploration/bin/python compute_run_metrics.py \
        --data_dir <sweep>/data/<sweep_id>/local --out <sweep>/analysis/run_metrics.json
"""

import argparse
import glob
import json
import os
import statistics


def reward_series(rows):
    """The logged reward series of one run: one (step, mean return) point per row."""
    # before: 611 train_history rows, each with a trailing 200-episode mean
    # after:  [(3276800, 0.0), (6553600, 400.0), ...] — the series everything else is derived from
    return [(r["step"], r["train/mean_extrinsic_reward"]) for r in rows
            if r.get("train/mean_extrinsic_reward") is not None]


def summarise_run(path):
    """Every summary this script knows how to compute, for one run's record."""
    with open(path) as f:
        rec = json.load(f)
    rows = rec.get("train_history") or []
    if not rows:
        return {"run_id": rec.get("run_id"), "error": "no logged rows"}

    last = rows[-1]
    series = reward_series(rows)
    values = [v for _, v in series]

    out = {
        "run_id": rec.get("run_id"),
        "arm": rec.get("arm"),
        "seed": rec.get("a_seed"),
        "completed": rec.get("completed"),
        "steps_reached": last["step"],
        "steps_target": rec.get("total_timesteps"),
        "fraction_of_target": last["step"] / rec["total_timesteps"] if rec.get("total_timesteps") else None,
        "episodes_finished": rec.get("episodes_seen", 0),
        "runtime_hours": rec.get("runtime_seconds", 0.0) / 3600,
        # CANDIDATE 1 — final performance: the last logged row.
        "final_mean_return_last_200_episodes": last.get("train/mean_extrinsic_reward"),
        "final_max_return_last_200_episodes": last.get("train/max_extrinsic_reward"),
        # CANDIDATE 2 — whole-run average: the mean of every logged row.
        "whole_run_mean_return": statistics.fmean(values) if values else None,
        "whole_run_median_return": statistics.median(values) if values else None,
        # The best row, and where it happened.
        "best_row_mean_return": max(values) if values else None,
        "best_row_step": max(series, key=lambda x: x[1])[0] if series else None,
        # The last quarter of the run, for a final number less noisy than a single row.
        "final_quarter_mean_return": (
            statistics.fmean(values[max(1, len(values) * 3 // 4) - 1:]) if values else None),
        "n_logged_rows": len(values),
    }
    # The gradient answer this run was launched to get, averaged over the run's logged rows.
    # "predictor clip fired" is the headline: how often the RND predictor's gradient was scaled down.
    # Under a joint clip that happens whenever the two networks' combined norm exceeds the threshold,
    # so it is not evidence the predictor caused the excess — the norm columns say who did.
    grad_rows = [r for r in (rec.get("eval_history") or []) if "grad/predictor_clip_fired_fraction" in r]
    if grad_rows:
        out.update({
            "mean_predictor_clip_fired_fraction": statistics.fmean(
                r["grad/predictor_clip_fired_fraction"] for r in grad_rows),
            "final_predictor_clip_fired_fraction": grad_rows[-1]["grad/predictor_clip_fired_fraction"],
            "mean_policy_clip_fired_fraction": statistics.fmean(
                r["grad/policy_clip_fired_fraction"] for r in grad_rows),
            "predictor_share_of_squared_norm": statistics.fmean(
                r["grad/predictor_share_of_squared_norm"] for r in grad_rows
                if r.get("grad/predictor_share_of_squared_norm") is not None),
            "mean_predictor_norm": statistics.fmean(
                r["grad/mean_predictor_norm_before_clipping"] for r in grad_rows),
            "mean_policy_norm": statistics.fmean(
                r["grad/mean_policy_norm_before_clipping"] for r in grad_rows),
            "max_joint_norm_seen": max(r["grad/max_joint_norm_before_clipping"] for r in grad_rows),
            "joint_grad_clip": grad_rows[-1].get("grad/joint_grad_clip"),
        })
    return out


def aggregate_by_arm(runs):
    """Group the per-run summaries by arm, reporting mean, median and spread across seeds."""
    by_arm = {}
    for r in runs:
        if "error" in r:
            continue
        by_arm.setdefault(r["arm"], []).append(r)
    out = {}
    for arm, rs in sorted(by_arm.items()):
        entry = {"seeds": len(rs),
                 "seeds_completed": sum(1 for r in rs if r["completed"]),
                 "mean_fraction_of_target": statistics.fmean(
                     r["fraction_of_target"] for r in rs if r["fraction_of_target"] is not None)}
        for metric in ["final_mean_return_last_200_episodes", "whole_run_mean_return",
                       "final_quarter_mean_return", "best_row_mean_return",
                       "mean_predictor_clip_fired_fraction", "predictor_share_of_squared_norm",
                       "mean_predictor_norm", "mean_policy_norm"]:
            vals = [r[metric] for r in rs if r.get(metric) is not None]
            if not vals:
                continue
            entry[metric] = {
                "mean": statistics.fmean(vals),
                # The median matters here: a sparse-reward Atari task is long-tailed across seeds,
                # so one lucky seed moves the mean a long way and the median does not follow it.
                "median": statistics.median(vals),
                "standard_deviation": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
                "min": min(vals), "max": max(vals), "n": len(vals),
            }
        out[arm] = entry
    return out


def matched_step_comparison(records, min_runs_per_arm=5):
    """Compare the arms only at steps every counted run has reached, and return the whole curve.

    This is the comparison that can be trusted while the campaign is unfinished, and this campaign
    will not finish: 150 runs of 2e9 steps is more compute than the cluster gives in one stretch. A
    run's last logged step says as much about the speed of the node it landed on as about its arm, so
    ranking arms by their final row would rank node allocations. Reading every arm at the same step
    removes that entirely.

    Returns one entry per step on the shared grid, so the arms can be plotted against each other over
    training rather than compared at a single point.
    """
    # before: runs with last steps [3.2M, 29M, 16M, 42M, ...] and rows on a common 3,276,800 grid
    # after:  one entry per grid step, holding only the runs that had REACHED that step
    by_arm = {}
    for r in records:
        by_arm.setdefault(r["arm"], []).append(r)
    all_steps = sorted({s for r in records for s, _ in r["_series"]})

    curve = []
    for step in all_steps:
        entry = {"step": step, "arms": {}}
        for arm, rs in sorted(by_arm.items()):
            # a run counts at this step only if it got there; its value is the best row up to it
            reached = [max((v for s, v in r["_series"] if s <= step), default=0.0)
                       for r in rs if r["_last_step"] >= step]
            if len(reached) < min_runs_per_arm:
                continue
            entry["arms"][arm] = {
                "runs_reaching_this_step": len(reached),
                "runs_that_have_scored": sum(1 for v in reached if v > 0),
                "fraction_scored": sum(1 for v in reached if v > 0) / len(reached),
                "mean": statistics.fmean(reached),
                "median": statistics.median(reached),
                "max": max(reached),
            }
        if len(entry["arms"]) == len(by_arm):
            curve.append(entry)
    return curve


def print_matched_step_table(curve):
    """Print the deepest step at which every arm still has enough runs, plus the trend before it."""
    if not curve:
        print("\nNo step has been reached by enough runs in every arm yet.")
        return
    last = curve[-1]
    print(f"\nEvery arm read at the same step — {last['step']:,}, the deepest point every arm still")
    print("has runs at. A run counts only if it reached that step; its value is its best row up to it.")
    hdr = f"{'arm':<26}{'runs':>6}{'scored':>8}{'rate':>8}{'mean':>10}{'median':>9}{'max':>10}"
    print(hdr); print("-" * len(hdr))
    for arm, a in last["arms"].items():
        print(f"{arm:<26}{a['runs_reaching_this_step']:>6}{a['runs_that_have_scored']:>8}"
              f"{100*a['fraction_scored']:>7.0f}%{a['mean']:>10.1f}{a['median']:>9.1f}{a['max']:>10.1f}")
    # the trend matters more than any single step while the runs are this early
    print(f"\nFraction of runs that have scored, over the shared grid "
          f"({len(curve)} step(s), every arm present):")
    arms = list(last["arms"])
    print("  " + f"{'step':>13}" + "".join(f"{a.split('_')[0]:>10}" for a in arms))
    for e in curve[-6:]:
        print("  " + f"{e['step']:>13,}" + "".join(f"{100*e['arms'][a]['fraction_scored']:>9.0f}%" for a in arms))


def main():
    """Summarise every record in a directory and write one JSON plus a printed table."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="directory holding <id>_of_<total>.json records")
    ap.add_argument("--out", required=True, help="where to write the summary JSON")
    ap.add_argument("--completed_only", action="store_true",
                    help="summarise only runs that reached their step target")
    args = ap.parse_args()

    paths = sorted(p for p in glob.glob(os.path.join(args.data_dir, "*_of_*.json"))
                   if not p.endswith(".episodes.jsonl"))
    runs = [summarise_run(p) for p in paths]
    if args.completed_only:
        runs = [r for r in runs if r.get("completed")]
    # The matched-step comparison needs each run's whole series, which summarise_run does not keep.
    for r, path in zip(runs, paths):
        rec = json.load(open(path))
        r["_series"] = reward_series(rec.get("train_history") or [])
        r["_last_step"] = r.get("steps_reached") or 0
    curve = matched_step_comparison([r for r in runs if "error" not in r and r["_series"]])
    for r in runs:
        r.pop("_series", None); r.pop("_last_step", None)
    report = {"data_dir": os.path.abspath(args.data_dir), "runs": runs,
              "by_arm": aggregate_by_arm(runs), "matched_step_curve": curve}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)
    os.chmod(args.out, 0o664)

    print(f"{len(runs)} runs from {args.data_dir}\n")
    hdr = (f"{'arm':<26}{'seeds':>6}{'done':>6}{'progress':>10}{'final':>10}"
           f"{'last 1/4':>10}{'whole-run':>11}{'best':>9}{'clip rate':>11}")
    print(hdr); print("-" * len(hdr))
    for arm, a in report["by_arm"].items():
        def g(m, k="median"):
            return f"{a[m][k]:.1f}" if m in a else "-"
        clip = (f"{a['mean_predictor_clip_fired_fraction']['median']:.3f}"
                if "mean_predictor_clip_fired_fraction" in a else "-")
        print(f"{arm:<26}{a['seeds']:>6}{a['seeds_completed']:>6}"
              f"{a['mean_fraction_of_target']*100:>9.1f}%"
              f"{g('final_mean_return_last_200_episodes'):>10}"
              f"{g('final_quarter_mean_return'):>10}"
              f"{g('whole_run_mean_return'):>11}{g('best_row_mean_return'):>9}{clip:>11}")
    print("\nvalues are medians across seeds; the JSON carries mean, median, spread, min and max")
    print_matched_step_table(report["matched_step_curve"])
    print(f"wrote {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
