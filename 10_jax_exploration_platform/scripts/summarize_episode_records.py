"""Summarize runs recorded with --diagnostics or --episode-log: per setting, the curve of mean
episode return, its peak, its final-10% level, and the goal-reaching share, from the per-episode
sums.

Every window record carries, per copy, the sums over the episodes that ENDED in that window: the
diagnostics' tracker fields (episode_episodes, episode_reached, episode_ret, ...) or the episode
log's (episodes_per_copy, arrivals_per_copy, return_sum_per_copy, ...). A copy's mean episode
return in a window is the return sum divided by the episodes; the setting's curve is the mean of
that over its copies. This works for continuing and terminating environments alike, because it
counts real episode ends. With the episode log, an episode "reached the goal" when it ended there,
the return is of the reward the learner sees, and the steps to the goal and each copy's first
training step with an arrival are summarized as well.

Definitions (per setting; c indexes copies, w windows, W the number of windows):
- curve[w] = mean over copies of ret[c, w] / episodes[c, w]
- peak = max over w of curve[w]; peak step = env steps per copy at that window
- final = mean over copies of (mean over the last ceil(W / 10) windows of the copy's return)
- whole run = mean over copies of (episode-weighted mean return over all windows)
- holds peak = final / peak
- reach share (final) = sum of reached / sum of episodes over the final windows, all copies
- ever reached = share of copies with at least one episode that reached the goal
Uncertainties are standard errors over copies. A terminating task with a negative per-step shift
(0 at the goal, -1 elsewhere) has every episode's return raised by -shift x cap, so it reads on the
same scale as the lump-sum payment and its peak and final can be compared (`return_offset`).

Usage: summarize_episode_records.py <run folder> [<run folder> ...] --output summary.json
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np


def read_shard(path: Path):
    """The latest complete-or-partial attempt of one shard: its unit_start and window records."""
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    # a re-run appends a fresh attempt after a partial one; keep the last attempt only
    starts = [i for i, r in enumerate(records) if r.get("record") == "unit_start"]
    if not starts:
        return None, [], False
    tail = records[starts[-1]:]
    windows = [r for r in tail if r.get("record") == "episode_window"]
    complete = any(r.get("record") == "unit_complete" for r in tail)
    return tail[0], windows, complete


def return_offset(environment: str) -> float:
    """What to add to each episode's return so that a terminating task with a negative step shift
    reads on the lump-sum scale: 0 for an episode that never reaches the goal, cap - k + 1 for one
    that reaches it at step k. Zero for every other environment.

    before: pointmaze_medium_term250_nonoise_start6-1_goal1-6_shift-1@1 -> 250 (= -shift x cap)
    before: pointmaze_medium_cont250_nonoise_start6-1_goal1-6@1 -> 0
    """
    name = environment.split("@")[0]
    kind_cap = name.split("_")[2]
    if not kind_cap.startswith("term") or "_shift" not in name:
        return 0.0
    shift = float(name.split("_shift")[1].split("_")[0])
    return -shift * float(kind_cap[len("term"):])


def summarize_cell(windows, copies, offset: float = 0.0):
    """The numbers of one setting from its copies' columns of every window (see module doc)."""
    # before: one record per window with per-copy lists; after: arrays [windows, copies]. The
    # diagnostics' tracker is read when a record has it, the episode log otherwise.
    tracked = "diagnostics_per_copy" in windows[-1]
    column = lambda name: np.array([[(w["diagnostics_per_copy"] if tracked else w)[name][c]
                                     for c in copies] for w in windows], dtype=float)
    names = (("episode_episodes", "episode_ret", "episode_reached") if tracked else
             ("episodes_per_copy", "return_sum_per_copy", "arrivals_per_copy"))
    episodes, ret, reached = (column(name) for name in names)
    ret = ret + offset * episodes
    steps = np.array([w["env_steps_per_copy"] for w in windows], dtype=float)
    # before: ret, episodes [W, copies]; after: per-copy per-window mean return, NaN where no
    # episode ended in that window
    with np.errstate(invalid="ignore", divide="ignore"):
        per_copy = np.where(episodes > 0, ret / episodes, np.nan)
    curve = np.nanmean(per_copy, axis=1)
    curve_se = np.nanstd(per_copy, axis=1, ddof=1) / np.sqrt(np.sum(~np.isnan(per_copy), axis=1))
    k = int(np.nanargmax(curve))
    tail = max(1, math.ceil(len(windows) / 10))
    final_per_copy = ret[-tail:].sum(0) / np.maximum(episodes[-tail:].sum(0), 1)
    whole_per_copy = ret.sum(0) / np.maximum(episodes.sum(0), 1)
    n = len(copies)
    se = lambda x: float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float("nan")
    out = {
        "copies": n, "windows": len(windows),
        "steps_per_copy": float(steps[-1]),
        "peak": float(curve[k]), "peak_se": float(curve_se[k]),
        "peak_step": float(steps[k]), "peak_fraction_of_run": float(steps[k] / steps[-1]),
        "final": float(final_per_copy.mean()), "final_se": se(final_per_copy),
        "whole_run": float(whole_per_copy.mean()), "whole_run_se": se(whole_per_copy),
        "holds_peak": float(final_per_copy.mean() / curve[k]) if curve[k] > 0 else float("nan"),
        "reach_share_final": float(reached[-tail:].sum() / max(episodes[-tail:].sum(), 1)),
        "reach_share_peak": float(reached[k].sum() / max(episodes[k].sum(), 1)),
        "ever_reached": float(np.mean(reached.sum(0) > 0)),
        "curve_steps": steps.tolist(), "curve": curve.tolist(), "curve_se": curve_se.tolist(),
        "reach_share_curve": (reached.sum(1) / np.maximum(episodes.sum(1), 1)).tolist(),
    }
    if "coverage_per_copy" in windows[-1]:
        cov = np.array([windows[-1]["coverage_per_copy"][c] for c in copies])
        out["coverage_final"], out["coverage_final_se"] = float(cov.mean()), se(cov)
    if not tracked:
        # the episode log's own numbers: episode length and steps to the goal over the final
        # windows, and when each copy first arrived (-1: never; counted as the whole run in the
        # mean capped first-goal step)
        lengths, to_goal = column("length_sum_per_copy"), column("steps_to_goal_sum_per_copy")
        first = np.array([windows[-1]["first_goal_step_per_copy"][c] for c in copies], dtype=float)
        out.update({
            "episode_length_final": float(lengths[-tail:].sum() / max(episodes[-tail:].sum(), 1)),
            "steps_to_goal_final": (float(to_goal[-tail:].sum() / reached[-tail:].sum())
                                    if reached[-tail:].sum() > 0 else float("nan")),
            "first_goal_step_per_copy": first.tolist(),
            "goal_discovery_rate": float(np.mean(first >= 0)),
            "mean_capped_first_goal_step": float(np.where(first >= 0, first, steps[-1]).mean())})
        return out
    # final-window means of every other diagnostic, so a mechanism can be read off the table
    tail_diag = {}
    for name in windows[-1]["diagnostics_per_copy"]:
        if name.startswith("episode_"):
            continue
        values = np.array([[w["diagnostics_per_copy"][name][c] for c in copies]
                           for w in windows[-tail:]], dtype=float)
        tail_diag[name] = float(values.mean())
        peak_values = np.array([windows[k]["diagnostics_per_copy"][name][c] for c in copies])
        tail_diag[name + "_at_peak"] = float(peak_values.mean())
    out["diagnostics_final_and_at_peak"] = tail_diag
    # copy-mean curves of the diagnostics that explain a fall, one value per window
    out["diagnostic_curves"] = {
        name: [float(np.mean([w["diagnostics_per_copy"][name][c] for c in copies]))
               for w in windows]
        for name in ("logstd_mean", "speed_mean", "ext_explained_variance", "adv_int_rms",
                     "adv_ext_rms", "grad_norm_critic")
        if name in windows[-1]["diagnostics_per_copy"]}
    return out


def cell_values(start: dict, setting: list) -> dict:
    """One cell's settings by name, with the learning rate and intrinsic weight always present.

    A unit written before cells had names holds (learning rate, intrinsic weight) pairs. A value
    the cell does not hold is the unit's own scalar.
    before: names ["int_coef", "ent_coef"], cell [4.0, 0.01], the unit's learning rate 1e-4
    after:  {"learning_rate": 1e-4, "int_coef": 4.0, "ent_coef": 0.01}
    """
    names = start.get("cell_setting_names", ["learning_rate", "int_coef"])
    values = dict(zip(names, setting))
    for name in ("learning_rate", "int_coef"):
        if name not in values:
            values[name] = start["trainer_config"][name]
    return values


def summarize_run(run: Path):
    """Every unit of one run folder, split into its settings."""
    result = {}
    for shard in sorted((run / "data").glob("*.jsonl")):
        start, windows, complete = read_shard(shard)
        if start is None or not windows or not (
                "diagnostics_per_copy" in windows[-1] or "episodes_per_copy" in windows[-1]):
            continue
        cells = start["cell_settings"]
        copy_cell = start["copy_cell"]
        for g, setting in enumerate(cells):
            copies = [c for c, cell in enumerate(copy_cell) if cell == g]
            values = cell_values(start, setting)
            # the key names the rate and the weight as it always did, then every other setting
            # the cell holds, so two cells that differ only there keep separate entries
            others = {name: value for name, value in values.items()
                      if name not in ("learning_rate", "int_coef")}
            key = (f"{start['unit_id']}|lr={values['learning_rate']:g}|beta={values['int_coef']:g}"
                   + "".join(f"|{name}={value:g}" for name, value in others.items()))
            offset = return_offset(start["environment"])
            result[key] = {"unit": start["unit_id"], "bonus": start["bonus"],
                           "environment": start["environment"], "return_offset": offset,
                           "learning_rate": values["learning_rate"],
                           "intrinsic_weight": values["int_coef"], "other_settings": others,
                           "complete": complete, **summarize_cell(windows, copies, offset)}
    return result


def main():
    """Summarize every given run folder into one JSON file."""
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = {}
    for run in args.runs:
        summary.update({f"{run.name}|{k}": v for k, v in summarize_run(run).items()})
    args.output.write_text(json.dumps(summary, indent=1) + "\n")
    for key, s in summary.items():
        print(f"{s['unit']:42s} beta={s['intrinsic_weight']:<7g} complete={s['complete']!s:5} "
              f"peak={s['peak']:.3f}@{s['peak_step'] / 1e6:.1f}M final={s['final']:.3f} "
              f"holds={s['holds_peak']:.2f} reach_final={s['reach_share_final']:.3f} "
              f"ever={s['ever_reached']:.2f}")


if __name__ == "__main__":
    main()
