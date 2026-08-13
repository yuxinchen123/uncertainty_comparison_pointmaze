#!/usr/bin/env python
"""Train run 6's writeup artifacts, regenerated from live records as the sweep advances:

- plots/reward_table.tex : the Table-55-like performance table — the run-5 original RND reference
  row (the frozen bar, final at n=300) plus one row per algorithm arm at its best configuration so
  far. An arm with no configuration at MIN_SEEDS completed seeds gets a "still running" row with
  its total finished-seed count, so the table never silently omits a running arm.
- plots/reward_curve.pdf/.png : the Figure-17-like curve — the run-5 original RND at its winning
  bonus weight as a BLACK DASHED line (from train run 5's own records, cached after the first
  read), and each algorithm arm's best configuration as a SOLID color line.

Both artifacts are safe to regenerate at any time; before the first records land they show the
reference alone.
"""
import glob
import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ANA = os.path.dirname(HERE)
RUN_DIR = os.path.dirname(ANA)
PLOTS = os.path.join(ANA, "plots")
DATA = os.path.join(ANA, "data")
sys.path.insert(0, os.path.join(RUN_DIR, "slurm"))
import build_queue as bq   # noqa: E402
import score_rules         # noqa: E402

MIN_SEEDS = 5   # below this an arm's best-so-far mean is too noisy to name
SUCCESS_THRESHOLD = 5.0   # run 5's succ. definition: final reward > 5

# the run-5 records that define the reference row and the black dashed curve
RUN5_LOCAL = os.path.join(
    os.path.dirname(RUN_DIR),
    "2026-07-20-16-44_run_5_baseline_rnd-next-state__origsmall-mse-mean-lr1e-4-out128-predextra1-"
    "leaky0.2-warmupenv6400-rewardnorm-beta1e-2to1e4+0.5-prune30__C2-adam-mse-b100__N1-rewardnorm-"
    "b1e4__seed600-899_1Mstep_cpu-nolim",
    "data", "2026-07-20-16-55_set-baseline", "local")
REF_CACHE = os.path.join(DATA, "run5_reference.json")
ARM_COLOR = {"alg1": "#0072B2", "alg2.1": "#009E73", "alg2.2": "#E69F00", "alg2.3": "#CC79A7"}
ARM_LABEL = {"alg1": "algorithm 1", "alg2.1": "algorithm 2.1",
             "alg2.2": "algorithm 2.2", "alg2.3": "algorithm 2.3"}


def mean_se(vals):
    """Sample mean and standard error of a non-empty list."""
    n = len(vals)
    m = sum(vals) / n
    if n < 2:
        return m, 0.0
    sd = math.sqrt(sum((x - m) ** 2 for x in vals) / (n - 1))
    return m, sd / math.sqrt(n)


def load_run5_reference():
    """(scores, curves) of the run-5 original RND winner (Adam 1e-4, bonus weight 1000), cached.

    before: 1,913 run-5 JSONs across three arms; after: {"scores": [38.4, ...300 floats],
    "curves": [[steps...], [means...]] x 300} for the beta=1000 original-small line only.
    """
    if os.path.exists(REF_CACHE):
        return json.load(open(REF_CACHE))
    out = {"scores": [], "curves": []}
    for path in glob.glob(os.path.join(RUN5_LOCAL, "*.json")):
        try:
            d = json.load(open(path))
        except (ValueError, OSError):
            continue
        if not d.get("completed", True):
            continue
        if d.get("rnd_bonus_readout") != "mse_mean":
            continue                              # the benchmark / reward-norm arms
        if abs(float(d.get("rnd_lr", 0)) - 1e-4) > 1e-12 or "%g" % float(d["beta"]) != "1000":
            continue
        rows = d.get("train_history") or []
        if not rows:
            continue
        out["scores"].append(rows[-1]["train/mean_extrinsic_reward"])
        out["curves"].append([[float(r["step"]) for r in rows],
                              [float(r["train/mean_extrinsic_reward"]) for r in rows]])
    os.makedirs(DATA, exist_ok=True)
    with open(REF_CACHE, "w") as fh:
        json.dump(out, fh)
    return out


def load_run6():
    """{arm: {beta_lr_key: {"scores": [...], "curves": [...]}}} over the 1M sweep's completed
    records (the 96-hour sweep shares the data/ tree, so its records are filtered out by their
    step budget and loaded separately by load_ext96h)."""
    out = {arm: {} for arm in bq.ARMS}
    for sweep_dir in glob.glob(os.path.join(RUN_DIR, "data", "*", "local")):
        for path in glob.glob(os.path.join(sweep_dir, "*.json")):
            try:
                d = json.load(open(path))
            except (ValueError, OSError):
                continue
            if not d.get("completed", True):
                continue
            if int(d.get("total_timesteps", 1000000)) != 1000000:
                continue
            rows = d.get("train_history") or []
            if not rows:
                continue
            arm = bq.arm_from_record(d)
            key = f'lr{"%g" % float(d["rnd_lr"])}_b{"%g" % float(d["beta"])}'
            e = out[arm].setdefault(key, {"scores": [], "curves": []})
            e["scores"].append(rows[-1]["train/mean_extrinsic_reward"])
            e["curves"].append([[float(r["step"]) for r in rows],
                                [float(r["train/mean_extrinsic_reward"]) for r in rows]])
    return out


# the 96-hour groups in table/plot order: dict key -> (row label, curve label, color, linestyle)
EXT96H_GROUPS = {
    "run5-origrnd": (r"run-5 original RND, 96 h (Adam $10^{-4}$, $\beta{=}1000$)",
                     "run-5 original RND, 96 h ($\\beta$=1000)", "black", "-"),
    "alg2.3": (r"algorithm 2.3, 96 h (lr $0.01$, $\beta{=}30$)",
               "algorithm 2.3, 96 h (lr=0.01 $\\beta$=30)", "#CC79A7", "-."),
    "gt": (r"ground-truth bonus $\min(1,1/\sqrt{n})$, 96 h ($\beta{=}1$)",
           "ground-truth bonus, 96 h ($\\beta$=1)", "#D55E00", "-"),
}
# the fixed milestones the 96-hour block reports (the user's choice 2026-08-13): a configuration
# fills a milestone row once >= MIN_SEEDS of its seeds logged that step
MILESTONES = [2000000, 4000000, 6000000]


def ext96h_group_of(d):
    """Which 96-hour group a record belongs to, from its own fields.
    before: {"algorithm": "gt_position_velocity", ...} / {"rnd_bonus_readout": "mse_mean", ...}
    after:  "gt" / "run5-origrnd" (everything else in this sweep is algorithm 2.3)."""
    if d.get("algorithm") == "gt_position_velocity":
        return "gt"
    if d.get("rnd_bonus_readout") == "mse_mean":
        return "run5-origrnd"
    return "alg2.3"


def load_ext96h():
    """{group: {"at": {milestone: [scores]}, "curves": [...], "n_done": int}} over the 96-hour
    sweep's completed records (runs end at their job's wall, so lengths differ per node; each
    record contributes to every milestone its curve reached), or None before the sweep exists."""
    sweep_dirs = glob.glob(os.path.join(RUN_DIR, "data", "*run6ext96h*", "local"))
    if not sweep_dirs:
        return None
    out = {g: {"at": {m: [] for m in MILESTONES}, "curves": [], "n_done": 0}
           for g in EXT96H_GROUPS}
    for sweep_dir in sweep_dirs:
        for path in glob.glob(os.path.join(sweep_dir, "*.json")):
            try:
                d = json.load(open(path))
            except (ValueError, OSError):
                continue
            if not d.get("completed", True):
                continue
            if int(d.get("total_timesteps", 0)) != 10000000:
                continue
            rows = d.get("train_history") or []
            if not rows:
                continue
            e = out[ext96h_group_of(d)]
            e["n_done"] += 1
            # before: rows=[{"step": 50000, "train/mean_extrinsic_reward": 1.2}, ...] up to the
            # run's walltime end; after: at[2000000] gains the value of the step==2000000 row
            by_step = {int(r["step"]): float(r["train/mean_extrinsic_reward"]) for r in rows}
            for m in MILESTONES:
                if m in by_step:
                    e["at"][m].append(by_step[m])
            e["curves"].append([[float(r["step"]) for r in rows],
                                [float(r["train/mean_extrinsic_reward"]) for r in rows]])
    return out


def best_of(arm_data):
    """(key, entry) with the best mean score among configs at MIN_SEEDS seeds, or (None, None)."""
    ranked = [(k, sum(e["scores"]) / len(e["scores"])) for k, e in arm_data.items()
              if len(e["scores"]) >= MIN_SEEDS]
    if not ranked:
        return None, None
    k = max(ranked, key=lambda t: t[1])[0]
    return k, arm_data[k]


def fmt(x, nd=2):
    """Format a float, or '--' for None/NaN."""
    return "--" if x is None or x != x else f"{x:.{nd}f}"


def write_table(ref, run6, ext96h):
    """The performance table: reference row + one row per arm (best config or still-running),
    then — once the 96-hour sweep exists — one row per (configuration, milestone) with >=
    MIN_SEEDS seeds past that step. Ranking (bold best / underline second) is separate for the
    1M block and for EACH milestone sub-block: different step budgets are never compared."""
    n_ref, (m_ref, se_ref) = len(ref["scores"]), mean_se(ref["scores"])
    succ_ref = sum(1 for s in ref["scores"] if s > SUCCESS_THRESHOLD) / n_ref
    rows = [[r"run-5 original RND, Adam $10^{-4}$ ($\beta{=}1000$) --- the frozen bar",
             m_ref, se_ref, succ_ref, n_ref]]
    for arm in bq.ARMS:
        key, e = best_of(run6[arm])
        if key is None:
            done = sum(len(v["scores"]) for v in run6[arm].values())
            rows.append([f"{ARM_LABEL[arm]} (still running)", None, None, None, done])
            continue
        lr, beta = key.replace("lr", "").split("_b")
        m, se = mean_se(e["scores"])
        succ = sum(1 for s in e["scores"] if s > SUCCESS_THRESHOLD) / len(e["scores"])
        rows.append([f"{ARM_LABEL[arm]} (lr ${lr}$, $\\beta{{=}}{beta}$)",
                     m, se, succ, len(e["scores"])])
    # the 96-hour block: one milestone sub-block after another, each ranked on its own
    blocks = [(0, len(rows))]
    if ext96h is not None:
        any_row = False
        for m_step in MILESTONES:
            lo = len(rows)
            for group, (row_label, _, _, _) in EXT96H_GROUPS.items():
                scores = ext96h[group]["at"][m_step]
                if len(scores) < MIN_SEEDS:
                    continue
                m, se = mean_se(scores)
                succ = sum(1 for s in scores if s > SUCCESS_THRESHOLD) / len(scores)
                rows.append([f"{row_label} --- at {m_step // 1000000}M", m, se, succ,
                             len(scores)])
                any_row = True
            if len(rows) > lo:
                blocks.append((lo, len(rows)))
        if not any_row:
            # nothing at any milestone yet: one still-running row per group with its done count
            lo = len(rows)
            for group, (row_label, _, _, _) in EXT96H_GROUPS.items():
                rows.append([f"{row_label} --- still running", None, None, None,
                             ext96h[group]["n_done"]])
            blocks.append((lo, len(rows)))

    def block_marks(col):
        """{row index: 'bold'|'under'} per metric column, ranked within each block."""
        marks = {}
        for lo, hi in blocks:
            order = sorted(((i, rows[i][col]) for i in range(lo, hi)
                            if rows[i][col] is not None), key=lambda t: t[1], reverse=True)
            if order:
                marks[order[0][0]] = "bold"
            if len(order) > 1:
                marks[order[1][0]] = "under"
        return marks

    marks, s_marks = block_marks(1), block_marks(3)

    def tex(v, mark):
        return {"bold": f"\\textbf{{{v}}}", "under": f"\\underline{{{v}}}"}.get(mark, v)

    lines = [r"\begin{tabular}{@{}>{\raggedright\arraybackslash}p{5.4cm} r r r r@{}}",
             r"\toprule",
             r"\textbf{Arm} & $\bar R$ & $\mathrm{SE}$ & succ. & $n$ \\", r"\midrule"]
    rule_rows = {1} | {lo for lo, _ in blocks[1:]}
    for i, r in enumerate(rows):
        if i in rule_rows:
            lines.append(r"\midrule")
        lines.append(f"{r[0]} & {tex(fmt(r[1]), marks.get(i))} & {fmt(r[2])} & "
                     f"{tex(fmt(r[3], 2), s_marks.get(i))} & {r[4]} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    os.makedirs(PLOTS, exist_ok=True)
    with open(os.path.join(PLOTS, "reward_table.tex"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


def curve_band(curves):
    """(grid, mean, se, n) over per-seed curves sharing the modal grid, or None."""
    if not curves:
        return None
    grid = curves[0][0]
    mat = np.array([m for s, m in curves if s == grid and None not in m])
    if mat.size == 0:
        return None
    mean = mat.mean(0)
    se = mat.std(0, ddof=1) / math.sqrt(mat.shape[0]) if mat.shape[0] > 1 else np.zeros_like(mean)
    return np.array(grid), mean, se, mat.shape[0]


def curve_band_varlen(curves, min_seeds):
    """(grid, mean, se, n_at_each_step) over VARIABLE-LENGTH per-seed curves on the 50k grid,
    truncated where fewer than min_seeds seeds still have data (96-hour runs end at their node's
    own walltime step).

    before: curves=[([50000, 100000, 150000], [1.0, 2.0, 3.0]), ([50000, 100000], [1.1, 2.1])],
            min_seeds=2
    after:  grid=[50000, 100000], mean=[1.05, 2.05], se=[...], counts=[2, 2] (150000 dropped)."""
    if not curves:
        return None
    by_step = {}
    for steps, vals in curves:
        for s, v in zip(steps, vals):
            by_step.setdefault(float(s), []).append(float(v))
    grid = sorted(s for s, vs in by_step.items() if len(vs) >= min_seeds)
    if not grid:
        return None
    mean = np.array([np.mean(by_step[s]) for s in grid])
    se = np.array([np.std(by_step[s], ddof=1) / math.sqrt(len(by_step[s]))
                   if len(by_step[s]) > 1 else 0.0 for s in grid])
    return np.array(grid), mean, se, len(curves)


def write_curve(ref, run6, ext96h):
    """The reward-over-training figure: the reference black dashed, the four 1M arms solid
    colors, and — once they have data — the three 4M lines (extending the x axis to 4e6)."""
    fig, ax = plt.subplots(figsize=(6, 4))
    band = curve_band(ref["curves"])
    if band is not None:
        grid, mean, se, n = band
        ax.plot(grid, mean, "k--", linewidth=1.8,
                label=f"run-5 original RND, Adam $10^{{-4}}$ ($\\beta$=1000) (n={n})")
        ax.fill_between(grid, mean - se, mean + se, alpha=0.12, color="black")
    for arm in bq.ARMS:
        key, e = best_of(run6[arm])
        if key is None:
            continue
        band = curve_band(e["curves"])
        if band is None:
            continue
        grid, mean, se, n = band
        lr, beta = key.replace("lr", "").split("_b")
        ax.plot(grid, mean, color=ARM_COLOR[arm], linewidth=1.8,
                label=f"{ARM_LABEL[arm]} lr={lr} $\\beta$={beta} (n={n})")
        ax.fill_between(grid, mean - se, mean + se, alpha=0.15, color=ARM_COLOR[arm])
    if ext96h is not None:
        for group, (_, curve_label, color, linestyle) in EXT96H_GROUPS.items():
            e = ext96h[group]
            if len(e["curves"]) < MIN_SEEDS:
                continue
            band = curve_band_varlen(e["curves"], MIN_SEEDS)
            if band is None:
                continue
            grid, mean, se, n = band
            ax.plot(grid, mean, color=color, linestyle=linestyle, linewidth=1.8,
                    label=f"{curve_label} (n={n})")
            ax.fill_between(grid, mean - se, mean + se, alpha=0.15, color=color)
    ax.set_xlabel("environment steps")
    ax.set_ylabel("training-episode extrinsic reward")
    ax.set_title("Train run 6 — the four algorithms vs the run-5 original RND\n(mean $\\pm$ SE)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(PLOTS, f"reward_curve.{ext}"), dpi=150)
    plt.close(fig)


def main():
    ref = load_run5_reference()
    run6 = load_run6()
    ext96h = load_ext96h()
    n6 = sum(len(e["scores"]) for a in run6.values() for e in a.values())
    n96 = 0 if ext96h is None else sum(e["n_done"] for e in ext96h.values())
    print(f"reference: {len(ref['scores'])} run-5 seeds; run-6 completed records: {n6}; "
          f"ext96h completed records: {n96}")
    write_table(ref, run6, ext96h)
    write_curve(ref, run6, ext96h)
    print(f"wrote reward_table.tex, reward_curve.pdf/.png into {PLOTS}")


if __name__ == "__main__":
    main()
