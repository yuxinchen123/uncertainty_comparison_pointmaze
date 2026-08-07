#!/usr/bin/env python
"""Generate the Train run 5 (set baseline) analysis artifacts for main.tex.

Outputs (under ../plots/, each a bare tabular \\input by main.tex, plus one figure):
- reward_table.tex        the three arms (run-5 original-small winning beta, benchmark, reward-norm) + old references
                          reference rows; final training-episode reward Rbar +- SE, success fraction, n.
- beta_sweep_table.tex    the 8 run-5 original-small betas: Rbar +- SE, n, and whether pruned.
- reward_curve.pdf        mean +- SE training-reward curve per arm over 1e6 steps.
- substitution_table.tex  + ../substitution.md : per old run (3.2.2/3.1.1 for the benchmark; 3.2.3/3.2.4 for the reward-norm)
                          a Welch t-test, Cohen's d, and KS statistic vs the run-5 fresh line for the
                          same config_key; pre-registered "substantially different iff p<0.01 and
                          |d|>0.3"; the runs that pass BOTH gates may be pooled (substituted).

Score = last train_history row's train/mean_extrinsic_reward, COMPLETED records only. The output JSON
carries the individual rnd_ fields (not the queue-level config_key), so keys are reconstructed the same
way build_queue.config_key / prune_controller.key_from_record do.

Run:  /p/rlprojects/RND/.venvs/exploration/bin/python make_results.py
"""
import glob
import json
import math
import os
import re
import sys

import numpy as np
from scipy import stats

# Fast field extraction: the records carry a LARGE train_episode_history, so json.load over thousands
# of files is far too slow (it times out). Every field we need sits in the small config / train_history
# parts, so pull them with regex and never parse the episode history. train/mean_extrinsic_reward
# appears ONLY in train_history rows (train_episode_history uses train/extrinsic_reward), so a findall
# gives the per-eval training-reward curve, last element = the final reward.
RE_COMPLETED = re.compile(r'"completed":\s*(true|false)')
RE_STEP_MEAN = re.compile(r'"step":\s*([\d.eE+-]+),\s*"train/mean_extrinsic_reward":\s*([-\d.eE+naN]+)')

HERE = os.path.dirname(os.path.abspath(__file__))
RUN_DIR = os.path.dirname(os.path.dirname(HERE))
TRAIN_RUNS = os.path.dirname(RUN_DIR)
PLOTS = os.path.join(os.path.dirname(HERE), "plots")
SUBSTITUTION_MD = os.path.join(os.path.dirname(HERE), "substitution.md")
SUCCESS_THRESHOLD = 5.0

# the active run-5 sweep (single sweep in this folder); read the newest local dir
SWEEP_LOCALS = sorted(glob.glob(os.path.join(RUN_DIR, "data", "*", "local")))
SWEEP_LOCAL = SWEEP_LOCALS[-1] if SWEEP_LOCALS else None

# old-run data dirs (globbed so a path rename does not break this); each maps a run label to its glob
def _first(pat):
    """First directory matching pat under train_runs, or None."""
    g = sorted(glob.glob(pat))
    return g[0] if g else None

OLD_RUNS = {
    # benchmark sources
    "benchmark (run-3.2.2 data)": _first(os.path.join(TRAIN_RUNS, "*run_3_2_2*", "data", "*", "local")),
    "benchmark (run-3.1.1 data)": _first(os.path.join(TRAIN_RUNS, "*run_3_1_1*", "data", "*", "local")),
    # reward-norm sources
    "reward-norm (run-3.2.3 data)": _first(os.path.join(TRAIN_RUNS, "*run_3_2_3*", "data", "*", "local")),
    "reward-norm (run-3.2.4 data)": _first(os.path.join(TRAIN_RUNS, "*run_3_2_4*", "data", "*", "local")),
}

# canonical keys (8-field, matching build_queue.config_key)
# The Adam 1e-3 addendum on this same PointMaze task: a SEPARATE run folder, loaded here so
# Tables 55/56 and the reward curve can show both learning rates side by side. Its records carry the
# identical 8-field key as the 1e-4 arm (the key does not include the learning rate), so they are
# loaded by their own reader and kept in a separate dict, never merged into run5.
LR1E3_LOCAL = _first(os.path.join(TRAIN_RUNS, "*run_5_8_1_2_addendum*", "data", "*", "local"))
LR1E3_ENV = "initial_single_large_pointmaze_max_400"
RE_ENV_SETUP = re.compile(r'"env_setup":\s*"([^"]*)"')
RE_RND_LR = re.compile(r'"rnd_lr":\s*([\d.eE+-]+)')
RE_BETA = re.compile(r'"beta":\s*([\d.eE+-]+)')


def load_lr1e3(local_dir):
    """COMPLETED PointMaze records of the Adam 1e-3 addendum -> {beta string: [final rewards]}.

    Parsed with json.load, NOT the head+tail regex path that load() uses. Verified 2026-08-06: on an
    addendum record (524 kB) the head+tail window ends inside train_history and its last match is
    step 400000, so the fast reader silently returns a mid-training reward instead of the final one
    (61.01 where the true final is 93.81). run-5's own records are smaller and read correctly, which
    is why the bug only shows on this arm. Correctness beats speed here: this is a few hundred files.
    before: 1800 addendum JSONs across three environments
    after : {"1000": [40.1, 39.4, ...], "30": [...], ...} for PointMaze only
    """
    out = {}
    if not local_dir:
        return out
    for path in glob.glob(os.path.join(local_dir, "*.json")):
        try:
            d = json.load(open(path))
        except (ValueError, OSError):
            continue
        if not d.get("completed") or d.get("env_setup") != LR1E3_ENV:
            continue
        if abs(float(d.get("rnd_lr", 0)) - 1e-3) > 1e-12:
            continue
        rows = d.get("train_history") or []
        if not rows:
            continue
        out.setdefault("%g" % float(d["beta"]), []).append(rows[-1]["train/mean_extrinsic_reward"])
    return out


def load_lr1e3_curves(local_dir, beta):
    """Per-seed (steps, means) curves of ONE addendum bonus weight on this environment."""
    curves = []
    if not (local_dir and beta):
        return curves
    for path in glob.glob(os.path.join(local_dir, "*.json")):
        try:
            d = json.load(open(path))
        except (ValueError, OSError):
            continue
        if not d.get("completed") or d.get("env_setup") != LR1E3_ENV:
            continue
        if abs(float(d.get("rnd_lr", 0)) - 1e-3) > 1e-12 or "%g" % float(d["beta"]) != beta:
            continue
        rows = d.get("train_history") or []
        if rows:
            curves.append((tuple(float(r["step"]) for r in rows),
                           [float(r["train/mean_extrinsic_reward"]) for r in rows]))
    return curves


def lr1e3_best(lr1e3, min_seeds=5):
    """(beta, (n, mean, se, succ)) of the addendum's best weight, or (None, None) if too few seeds."""
    ranked = [(b, agg(v)) for b, v in lr1e3.items() if len(v) >= min_seeds]
    if not ranked:
        return None, None
    return max(ranked, key=lambda t: t[1][1])


KEY_C2 = "adam|mse|-|-|100|zero|-|-"
KEY_N1 = "adam|mse|-|-|10000|zero|rewardnorm|-"
# Display names for the two learning rates of the original-RND arm. The old name "original-small"
# was ambiguous (it named the network size, not the knob under test) and never said which learning
# rate a row or curve belonged to; both names now carry the rate explicitly.
LABEL_LR4 = "run-5 original RND, Adam $10^{-4}$"
LABEL_LR3 = "run-5 original RND, Adam $10^{-3}$"
BETAS = ["0.01", "0.1", "0.5", "1", "10", "100", "1000", "10000"]
ORIGSMALL_KEYS = [f"adam|mse_mean|-|-|{'%g' % float(b)}|zero|rewardnorm|origsmall" for b in BETAS]


def _g(pat, txt):
    """First regex group in txt, or None."""
    m = re.search(pat, txt)
    return m.group(1) if m else None


def key_from_text(txt):
    """Reconstruct the 8-field config_key from a record's raw TEXT (regex; same key the controller
    and build_queue produce). readout=='mse_mean' -> variant origsmall."""
    opt = _g(r'"rnd_optimizer":\s*"([^"]+)"', txt) or "adam"
    readout = _g(r'"rnd_bonus_readout":\s*"([^"]+)"', txt) or "mse"
    if opt == "sgd1t":
        eta0 = "%g" % float(_g(r'"rnd_sgd_eta0":\s*([-\d.eE+]+)', txt))
        t0 = "%g" % float(_g(r'"rnd_sgd_t0":\s*([-\d.eE+]+)', txt))
    else:
        eta0, t0 = "-", "-"
    beta = "%g" % float(_g(r'"beta":\s*([-\d.eE+]+)', txt))
    bias = _g(r'"rnd_bias_init":\s*"([^"]+)"', txt) or "zero"
    norm = "rewardnorm" if _g(r'"rnd_reward_norm":\s*(true|false)', txt) == "true" else "-"
    variant = "origsmall" if readout == "mse_mean" else "-"
    return f"{opt}|{readout}|{eta0}|{t0}|{beta}|{bias}|{norm}|{variant}"


def _head_tail(path, head=8192, tail=4096):
    """Read only the first `head` and last `tail` bytes of a file, concatenated. The record layout is
    [config + train_history  (first ~3 KB)] [train_episode_history  (huge middle, skipped)]
    [distance_history + rnd_ fields  (last few hundred bytes)], so head+tail carries everything we need
    (completed, beta, the train_history curve, and the rnd_ key fields) without reading the 150 KB
    episode-history middle. Falls back to the whole file when it is smaller than head+tail."""
    size = os.path.getsize(path)
    with open(path, "rb") as fh:
        if size <= head + tail:
            return fh.read().decode("utf-8", "replace")
        h = fh.read(head)
        fh.seek(size - tail)
        t = fh.read(tail)
    return h.decode("utf-8", "replace") + "\n" + t.decode("utf-8", "replace")


def load(local_dir, want_curve=False):
    """Load COMPLETED records from a local dir -> {key: {'final': [rewards], 'curves': [(steps, means)]}}
    using fast head+tail regex extraction (never reads the large train_episode_history). 'final' is the
    last train_history mean_extrinsic_reward; 'curves' (only if want_curve) is the per-eval (step, mean)
    sequence per seed for the reward-over-training plot."""
    out = {}
    if not local_dir:
        return out
    for path in glob.glob(os.path.join(local_dir, "*.json")):
        try:
            txt = _head_tail(path)
        except OSError:
            continue
        c = RE_COMPLETED.search(txt)
        if not c or c.group(1) != "true":
            continue
        pairs = RE_STEP_MEAN.findall(txt)   # (step, mean) for every train_history row, in order
        if not pairs:
            continue
        try:
            steps = [float(s) for s, _ in pairs]
            means = [float(m) for _, m in pairs]
        except ValueError:
            continue
        key = key_from_text(txt)
        rec = out.setdefault(key, {"final": [], "curves": []})
        rec["final"].append(means[-1])
        if want_curve:
            rec["curves"].append((tuple(steps), means))
    return out


def agg(vals):
    """(n, mean, se, success_fraction) for a list of final rewards."""
    n = len(vals)
    if n == 0:
        return 0, float("nan"), float("nan"), float("nan")
    mean = float(np.mean(vals))
    se = float(np.std(vals, ddof=1) / math.sqrt(n)) if n > 1 else 0.0
    succ = float(np.mean([v > SUCCESS_THRESHOLD for v in vals]))
    return n, mean, se, succ


def fmt(x, nd=2):
    """Format a float, or '--' for NaN."""
    return "--" if x != x else f"{x:.{nd}f}"


def mark_rows(rows, value_idx):
    """Return a set of (row, 'bold'|'under') marks for the best/second-best in column value_idx.
    rows: list of tuples; value_idx: index of the numeric value (higher is better; NaN ignored)."""
    vals = [(i, r[value_idx]) for i, r in enumerate(rows) if r[value_idx] == r[value_idx]]
    order = sorted(vals, key=lambda t: t[1], reverse=True)
    marks = {}
    if order:
        marks[order[0][0]] = "bold"
    if len(order) > 1:
        marks[order[1][0]] = "under"
    return marks


def tex(v, mark):
    """Wrap a cell string per its mark (bold / underline / none)."""
    if mark == "bold":
        return f"\\textbf{{{v}}}"
    if mark == "under":
        return f"\\underline{{{v}}}"
    return v


# ---------------------------------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------------------------------

def write_reward_table(run5, olds, lr1e3=None):
    """The headline performance table: the original-RND arm at both learning rates (each at its own
    best bonus weight), the benchmark, the reward-norm arm, and the old reference rows."""
    # pick the winning run-5 original-small beta by mean over its finished seeds
    os_stats = [(b, agg(run5.get(k, {"final": []})["final"]))
                for b, k in zip(BETAS, ORIGSMALL_KEYS)]
    os_ranked = sorted(os_stats, key=lambda t: (-1 if t[1][1] != t[1][1] else t[1][1]), reverse=True)
    best_beta, best_agg = os_ranked[0]
    rows = []  # (label, n, mean, se, succ, source)
    n, m, se, sc = best_agg
    rows.append([f"{LABEL_LR4} ($\\beta{{=}}{best_beta}$)", n, m, se, sc, "run 5"])
    # the same stack at 1e-3, at ITS best weight: one extra row, right under its 1e-4 counterpart
    lr3_beta, lr3_agg = lr1e3_best(lr1e3 or {})
    if lr3_beta is not None:
        n3, m3, se3, sc3 = lr3_agg
        rows.append([f"{LABEL_LR3} ($\\beta{{=}}{lr3_beta}$)", n3, m3, se3, sc3, "addendum"])
    for label, key in (("run-3.2.2 benchmark (Adam, no norm)", KEY_C2), ("run-3.2.3 reward-norm (Adam)", KEY_N1)):
        n, m, se, sc = agg(run5.get(key, {"final": []})["final"])
        rows.append([label, n, m, se, sc, "run 5"])
    # reference rows: old benchmark (3.2.2) and old reward-norm (3.2.4)
    for label, run_label, key in (("benchmark (run-3.2.2 data)", "benchmark (run-3.2.2 data)", KEY_C2),
                                  ("reward-norm (run-3.2.4 data)", "reward-norm (run-3.2.4 data)", KEY_N1)):
        n, m, se, sc = agg(olds.get(run_label, {}).get(key, {"final": []})["final"])
        rows.append([label, n, m, se, sc, "prior"])
    # mark only the settled rows: the addendum row has a fraction of their seeds, so a leading point
    # estimate there would not mean a lead (its standard error is several times theirs)
    settled = [i for i, r in enumerate(rows) if r[5] != "addendum"]
    marks_r = {settled[j]: v for j, v in mark_rows([(rows[i][0], rows[i][2]) for i in settled], 1).items()}
    marks_s = {settled[j]: v for j, v in mark_rows([(rows[i][0], rows[i][4]) for i in settled], 1).items()}
    lines = [r"\begin{tabular}{@{}>{\raggedright\arraybackslash}p{4.6cm} r r r r@{}}", r"\toprule",
             r"\textbf{Arm} & $\bar R$ & $\mathrm{SE}$ & succ. & $n$ \\", r"\midrule"]
    n_this_run = sum(1 for r in rows if r[5] != "prior")
    for i, r in enumerate(rows):
        if i == n_this_run:
            lines.append(r"\midrule")
        rbar = tex(fmt(r[2]), marks_r.get(i))
        succ = tex(fmt(r[4], 2), marks_s.get(i))
        lines.append(f"{r[0]} & {rbar} & {fmt(r[3])} & {succ} & {r[1]} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    _write("reward_table.tex", "\n".join(lines))
    return best_beta


def write_beta_sweep_table(run5, pruned_keys, lr1e3=None):
    """Per-weight results at BOTH learning rates, side by side.

    Left block: the 1e-4 sweep's eight weights with the race verdict that stopped each one. Right
    block: the same weight at 1e-3 where the two grids share it. The 1e-3 grid runs 1e-3..1e4 at 1
    and 3 per decade and does not contain 0.5, so that one row reads N/A rather than a blank.
    """
    lr1e3 = lr1e3 or {}
    lines = [r"\begin{tabular}{@{}r r r r c c r r r@{}}", r"\toprule",
             r"& \multicolumn{4}{c}{\textbf{Adam $10^{-4}$}} & & "
             r"\multicolumn{3}{c}{\textbf{Adam $10^{-3}$}} \\",
             r"\cmidrule(lr){2-5}\cmidrule(lr){7-9}",
             r"$\beta$ & $\bar R$ & $\mathrm{SE}$ & $n$ & stopped & & $\bar R$ & $\mathrm{SE}$ & $n$ \\",
             r"\midrule"]
    for b, k in zip(BETAS, ORIGSMALL_KEYS):
        n, m, se, _ = agg(run5.get(k, {"final": []})["final"])
        pr = "yes" if k in pruned_keys else "no"
        label = (f"$10^{{{int(round(math.log10(float(b))))}}}$"
                 if float(b) not in (0.5,) else "$0.5$")
        vals = lr1e3.get(b)
        if vals:
            n3, m3, se3, _ = agg(vals)
            right = f"{fmt(m3)} & {fmt(se3)} & {n3}"
        else:
            right = r"\multicolumn{3}{c}{N/A --- not in the $10^{-3}$ grid}"
        lines.append(f"{label} & {fmt(m)} & {fmt(se)} & {n} & {pr} & & {right} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    _write("beta_sweep_table.tex", "\n".join(lines))


def cohen_d(a, b):
    """Cohen's d with pooled SD (0 if degenerate)."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    sp = math.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    return (np.mean(a) - np.mean(b)) / sp if sp > 0 else float("nan")


def write_substitution(run5, olds):
    """Per old run, test the run-5 fresh line vs the old line for the same config_key."""
    # tests: run-5 benchmark vs old benchmark sources; run-5 reward-norm vs old reward-norm sources
    cases = [("benchmark (run-3.2.2 data)", KEY_C2, "benchmark"), ("benchmark (run-3.1.1 data)", KEY_C2, "benchmark"),
             ("reward-norm (run-3.2.3 data)", KEY_N1, "reward-norm"), ("reward-norm (run-3.2.4 data)", KEY_N1, "reward-norm")]
    rows, md = [], ["# Train run 5 — per-run substitution test\n",
                    "Pre-registered rule: an old run is *substantially different* from the run-5 fresh line "
                    "(and so is NOT pooled) iff **p < 0.01 AND |d| > 0.3** (Welch t on the final "
                    "training-episode reward). A run that passes BOTH gates (not substantially different) "
                    "may be pooled after an executed-path code-diff audit.\n"]
    for old_label, key, arm in cases:
        new = run5.get(key, {"final": []})["final"]
        old = olds.get(old_label, {}).get(key, {"final": []})["final"]
        if len(new) < 2 or len(old) < 2:
            rows.append([old_label, len(old), len(new), "--", "--", "--", "insufficient data"])
            md.append(f"- **{old_label}** vs run-5 {arm}: insufficient data (n_old={len(old)}, n_new={len(new)}).")
            continue
        t, p = stats.ttest_ind(new, old, equal_var=False)
        d = cohen_d(new, old)
        ks, ksp = stats.ks_2samp(new, old)
        substantial = (p < 0.01) and (abs(d) > 0.3)
        verdict = "differs (keep separate)" if substantial else "poolable (pending code-diff)"
        rows.append([old_label, len(old), len(new), f"{p:.3g}", f"{d:.2f}", f"{ks:.2f}", verdict])
        md.append(f"- **{old_label}** vs run-5 {arm}: mean_old={np.mean(old):.2f} (n={len(old)}), "
                  f"mean_new={np.mean(new):.2f} (n={len(new)}); Welch p={p:.3g}, Cohen d={d:.2f}, "
                  f"KS={ks:.2f} -> **{verdict}**.")
    lines = [r"\begin{tabular}{@{}>{\raggedright\arraybackslash}p{3.0cm} r r r r r >{\raggedright\arraybackslash}p{3.4cm}@{}}",
             r"\toprule",
             r"Old run & $n_{\mathrm{old}}$ & $n_{\mathrm{new}}$ & $p$ & $d$ & KS & verdict \\", r"\midrule"]
    for r in rows:
        lines.append(" & ".join(str(x) for x in r) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    _write("substitution_table.tex", "\n".join(lines))
    with open(SUBSTITUTION_MD, "w") as fh:
        fh.write("\n".join(md) + "\n")


def _curve_matrix(curves):
    """(grid, mean, se, n) over the per-seed curves that share the modal eval grid, or None."""
    if not curves:
        return None
    grid = curves[0][0]
    mat = np.array([m for s, m in curves if s == grid and None not in m])
    if mat.size == 0:
        return None
    mean = mat.mean(0)
    se = mat.std(0, ddof=1) / math.sqrt(mat.shape[0]) if mat.shape[0] > 1 else np.zeros_like(mean)
    return np.array(grid), mean, se, mat.shape[0]


def write_reward_curve(run5, best_beta, lr1e3_curves=None, lr1e3_beta=None):
    """Mean +- SE training-reward curve per arm over the eval grid.

    Four curves: the original-RND arm at BOTH learning rates, each at its own best bonus weight, plus
    the benchmark and the reward-norm arms. Every legend entry names its learning rate, so no curve
    can be read as "the original RND" without knowing which rate produced it.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # mathtext labels so the legend prints a real beta, e.g. "... Adam $10^{-4}$ ($\beta$=1000)"
    arms = [(f"run-5 original RND, Adam $10^{{-4}}$ ($\\beta$={best_beta})",
             ORIGSMALL_KEYS[BETAS.index(best_beta)], "tab:blue"),
            ("run-3.2.2 benchmark (Adam $10^{-3}$)", KEY_C2, "tab:orange"),
            ("run-3.2.3 reward-norm (Adam $10^{-3}$)", KEY_N1, "tab:green")]
    fig, ax = plt.subplots(figsize=(6, 4))
    # the addendum arm, drawn dashed so it reads as the same stack at a different rate
    if lr1e3_curves and lr1e3_beta:
        mat = _curve_matrix(lr1e3_curves)
        if mat is not None:
            grid, mean, se, k = mat
            ax.plot(grid, mean, linestyle="--", color="tab:red",
                    label=f"run-5 original RND, Adam $10^{{-3}}$ ($\\beta$={lr1e3_beta}) (n={k})")
            ax.fill_between(grid, mean - se, mean + se, alpha=0.2, color="tab:red")
    for label, key, color in arms:
        curves = run5.get(key, {"curves": []})["curves"]
        if not curves:
            continue
        # align on the common eval grid (use the modal step vector)
        grid = curves[0][0]
        mat = np.array([m for s, m in curves if s == grid and None not in m])
        if mat.size == 0:
            continue
        mean = mat.mean(0)
        se = mat.std(0, ddof=1) / math.sqrt(mat.shape[0]) if mat.shape[0] > 1 else np.zeros_like(mean)
        ax.plot(grid, mean, label=f"{label} (n={mat.shape[0]})", color=color)
        ax.fill_between(grid, mean - se, mean + se, alpha=0.2, color=color)
    ax.set_xlabel("environment steps")
    ax.set_ylabel("training-episode extrinsic reward")
    # two lines: on one line this title runs past the right edge of a 6-inch figure
    ax.set_title("Train run 5 — reward over training, both learning rates\n"
                 "(mean $\\pm$ SE)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "reward_curve.pdf"))
    plt.close(fig)


def _write(name, content):
    """Write a plots/ artifact, creating the dir."""
    os.makedirs(PLOTS, exist_ok=True)
    with open(os.path.join(PLOTS, name), "w") as fh:
        fh.write(content + "\n")


def load_pruned_keys():
    """The set of run-5 original-small keys the controller pruned (from the newest decisions ledger)."""
    ledgers = sorted(glob.glob(os.path.join(RUN_DIR, "slurm", "prune_decisions_*.jsonl")))
    pruned = set()
    if ledgers:
        for line in open(ledgers[-1]):
            try:
                pruned.update(json.loads(line).get("pruned_this_cycle", []))
            except json.JSONDecodeError:
                continue
    return pruned


def main():
    """Load run-5 + old-run data, write every table, the substitution note, and the reward curve."""
    if SWEEP_LOCAL is None:
        sys.exit("no run-5 sweep data yet")
    run5 = load(SWEEP_LOCAL, want_curve=True)
    olds = {label: load(d) for label, d in OLD_RUNS.items() if d}
    n_completed = sum(len(v["final"]) for v in run5.values())
    print(f"run-5 completed records: {n_completed}; old runs loaded: {list(olds)}")
    lr1e3 = load_lr1e3(LR1E3_LOCAL)
    print(f"addendum (Adam 1e-3) PointMaze records: {sum(len(v) for v in lr1e3.values())} "
          f"across {len(lr1e3)} bonus weights")
    best_beta = write_reward_table(run5, olds, lr1e3)
    write_beta_sweep_table(run5, load_pruned_keys(), lr1e3)
    write_substitution(run5, olds)
    lr3_beta, _ = lr1e3_best(lr1e3)
    lr3_curves = load_lr1e3_curves(LR1E3_LOCAL, lr3_beta) if lr3_beta else None
    try:
        write_reward_curve(run5, best_beta, lr3_curves, lr3_beta)
    except Exception as e:  # a plotting failure must not lose the tables
        print(f"reward_curve skipped: {e}")
    print(f"wrote reward_table.tex, beta_sweep_table.tex, substitution_table.tex, "
          f"substitution.md, reward_curve.pdf into {PLOTS}")


if __name__ == "__main__":
    main()
