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
    records (the ext4m sweep shares the data/ tree, so the 4M records are filtered out by their
    step budget and loaded separately by load_ext4m)."""
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


# the ext4m groups in table/plot order: dict key -> (row label, curve label, color, linestyle)
EXT4M_GROUPS = {
    "run5-origrnd": (r"run-5 original RND, 4M steps (Adam $10^{-4}$, $\beta{=}1000$)",
                     "run-5 original RND, 4M ($\\beta$=1000)", "black", "-"),
    "alg2.3": (r"algorithm 2.3, 4M steps (lr $0.01$, $\beta{=}30$)",
               "algorithm 2.3, 4M (lr=0.01 $\\beta$=30)", "#CC79A7", "-."),
    "gt": (r"ground-truth bonus $\min(1,1/\sqrt{n})$, 4M steps ($\beta{=}1$)",
           "ground-truth bonus, 4M ($\\beta$=1)", "#D55E00", "-"),
}


def ext4m_group_of(d):
    """Which ext4m group a 4M record belongs to, from its own fields.
    before: {"algorithm": "gt_position_velocity", ...} / {"rnd_bonus_readout": "mse_mean", ...}
    after:  "gt" / "run5-origrnd" (everything else in this sweep is algorithm 2.3)."""
    if d.get("algorithm") == "gt_position_velocity":
        return "gt"
    if d.get("rnd_bonus_readout") == "mse_mean":
        return "run5-origrnd"
    return "alg2.3"


def load_ext4m():
    """{group: {"scores": [...], "curves": [...]}} over the 4M extension's completed records, or
    None when the ext4m sweep has not been built yet (pre-launch regenerations are unchanged)."""
    sweep_dirs = glob.glob(os.path.join(RUN_DIR, "data", "*run6ext4m*", "local"))
    if not sweep_dirs:
        return None
    out = {g: {"scores": [], "curves": []} for g in EXT4M_GROUPS}
    for sweep_dir in sweep_dirs:
        for path in glob.glob(os.path.join(sweep_dir, "*.json")):
            try:
                d = json.load(open(path))
            except (ValueError, OSError):
                continue
            if not d.get("completed", True):
                continue
            if int(d.get("total_timesteps", 0)) != 4000000:
                continue
            rows = d.get("train_history") or []
            if not rows:
                continue
            e = out[ext4m_group_of(d)]
            e["scores"].append(rows[-1]["train/mean_extrinsic_reward"])
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


def write_table(ref, run6, ext4m):
    """The performance table: reference row + one row per arm (best config or still-running),
    then — once the ext4m sweep exists — one row per 4M configuration. The 1M and 4M blocks are
    ranked separately (bold best / underline second per block): a 4M budget beating a 1M budget
    is not a finding."""
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
    # the 4M block, appended after the 1M arms (row index >= n_1m ranks separately)
    n_1m = len(rows)
    if ext4m is not None:
        for group, (row_label, _, _, _) in EXT4M_GROUPS.items():
            e = ext4m[group]
            if len(e["scores"]) < MIN_SEEDS:
                rows.append([f"{row_label} --- still running", None, None, None,
                             len(e["scores"])])
                continue
            m, se = mean_se(e["scores"])
            succ = sum(1 for s in e["scores"] if s > SUCCESS_THRESHOLD) / len(e["scores"])
            rows.append([row_label, m, se, succ, len(e["scores"])])

    def block_marks(col):
        """{row index: 'bold'|'under'} per metric column, ranked within the 1M and 4M blocks."""
        marks = {}
        for lo, hi in ((0, n_1m), (n_1m, len(rows))):
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
    for i, r in enumerate(rows):
        if i in (1, n_1m):
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


def write_curve(ref, run6, ext4m):
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
    if ext4m is not None:
        for group, (_, curve_label, color, linestyle) in EXT4M_GROUPS.items():
            e = ext4m[group]
            if len(e["scores"]) < MIN_SEEDS:
                continue
            band = curve_band(e["curves"])
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
    ext4m = load_ext4m()
    n6 = sum(len(e["scores"]) for a in run6.values() for e in a.values())
    n4 = 0 if ext4m is None else sum(len(e["scores"]) for e in ext4m.values())
    print(f"reference: {len(ref['scores'])} run-5 seeds; run-6 completed records: {n6}; "
          f"ext4m completed records: {n4}")
    write_table(ref, run6, ext4m)
    write_curve(ref, run6, ext4m)
    print(f"wrote reward_table.tex, reward_curve.pdf/.png into {PLOTS}")


if __name__ == "__main__":
    main()
