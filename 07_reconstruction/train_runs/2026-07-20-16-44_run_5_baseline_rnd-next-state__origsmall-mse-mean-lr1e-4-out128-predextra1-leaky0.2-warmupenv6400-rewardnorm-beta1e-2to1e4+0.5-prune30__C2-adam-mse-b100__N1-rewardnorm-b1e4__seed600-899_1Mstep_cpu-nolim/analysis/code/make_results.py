#!/usr/bin/env python
"""Generate the Train run 5 (set baseline) analysis artifacts for main.tex.

Outputs (under ../plots/, each a bare tabular \\input by main.tex, plus one figure):
- reward_table.tex        the three arms (original-small winning beta, C2, N1) + old-C2/old-N1
                          reference rows; final training-episode reward Rbar +- SE, success fraction, n.
- beta_sweep_table.tex    the 8 original-small betas: Rbar +- SE, n, and whether pruned.
- reward_curve.pdf        mean +- SE training-reward curve per arm over 1e6 steps.
- substitution_table.tex  + ../substitution.md : per old run (3.2.2/3.1.1 for C2; 3.2.3/3.2.4 for N1)
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
    # C2 sources
    "C2 (run 3.2.2)": _first(os.path.join(TRAIN_RUNS, "*run_3_2_2*", "data", "*", "local")),
    "C2 (run 3.1.1)": _first(os.path.join(TRAIN_RUNS, "*run_3_1_1*", "data", "*", "local")),
    # N1 sources
    "N1 (run 3.2.3)": _first(os.path.join(TRAIN_RUNS, "*run_3_2_3*", "data", "*", "local")),
    "N1 (run 3.2.4)": _first(os.path.join(TRAIN_RUNS, "*run_3_2_4*", "data", "*", "local")),
}

# canonical keys (8-field, matching build_queue.config_key)
KEY_C2 = "adam|mse|-|-|100|zero|-|-"
KEY_N1 = "adam|mse|-|-|10000|zero|rewardnorm|-"
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

def write_reward_table(run5, olds):
    """The headline performance table: original-small (best beta), C2, N1, + old references."""
    # pick the winning original-small beta by mean over its finished seeds
    os_stats = [(b, agg(run5.get(k, {"final": []})["final"]))
                for b, k in zip(BETAS, ORIGSMALL_KEYS)]
    os_ranked = sorted(os_stats, key=lambda t: (-1 if t[1][1] != t[1][1] else t[1][1]), reverse=True)
    best_beta, best_agg = os_ranked[0]
    rows = []  # (label, n, mean, se, succ, source)
    n, m, se, sc = best_agg
    rows.append([f"original-small ($\\beta{{=}}{best_beta}$)", n, m, se, sc, "run 5"])
    for label, key in (("C2 (Adam, no norm)", KEY_C2), ("N1 (Adam, reward norm)", KEY_N1)):
        n, m, se, sc = agg(run5.get(key, {"final": []})["final"])
        rows.append([label, n, m, se, sc, "run 5"])
    # reference rows: old C2 (3.2.2) and old N1 (3.2.4)
    for label, run_label, key in (("C2 (run 3.2.2)", "C2 (run 3.2.2)", KEY_C2),
                                  ("N1 (run 3.2.4)", "N1 (run 3.2.4)", KEY_N1)):
        n, m, se, sc = agg(olds.get(run_label, {}).get(key, {"final": []})["final"])
        rows.append([label, n, m, se, sc, "prior"])
    marks_r = mark_rows([(r[0], r[2]) for r in rows], 1)
    marks_s = mark_rows([(r[0], r[4]) for r in rows], 1)
    lines = [r"\begin{tabular}{@{}>{\raggedright\arraybackslash}p{4.6cm} r r r r@{}}", r"\toprule",
             r"\textbf{Arm} & $\bar R$ & $\mathrm{SE}$ & succ. & $n$ \\", r"\midrule"]
    for i, r in enumerate(rows):
        if i == 3:
            lines.append(r"\midrule")
        rbar = tex(fmt(r[2]), marks_r.get(i))
        succ = tex(fmt(r[4], 2), marks_s.get(i))
        lines.append(f"{r[0]} & {rbar} & {fmt(r[3])} & {succ} & {r[1]} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    _write("reward_table.tex", "\n".join(lines))
    return best_beta


def write_beta_sweep_table(run5, pruned_keys):
    """Per-beta original-small results (mean +- SE, n, pruned flag)."""
    lines = [r"\begin{tabular}{@{}r r r r c@{}}", r"\toprule",
             r"$\beta$ & $\bar R$ & $\mathrm{SE}$ & $n$ & pruned \\", r"\midrule"]
    for b, k in zip(BETAS, ORIGSMALL_KEYS):
        n, m, se, _ = agg(run5.get(k, {"final": []})["final"])
        pr = "yes" if k in pruned_keys else "no"
        lines.append(f"$10^{{{int(round(math.log10(float(b))))}}}$" if float(b) not in (0.5,) else "$0.5$")
        lines[-1] += f" & {fmt(m)} & {fmt(se)} & {n} & {pr} \\\\"
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
    # tests: run-5 C2 vs old-C2 sources; run-5 N1 vs old-N1 sources
    cases = [("C2 (run 3.2.2)", KEY_C2, "C2"), ("C2 (run 3.1.1)", KEY_C2, "C2"),
             ("N1 (run 3.2.3)", KEY_N1, "N1"), ("N1 (run 3.2.4)", KEY_N1, "N1")]
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


def write_reward_curve(run5, best_beta):
    """Mean +- SE training-reward curve per arm over the eval grid."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # mathtext label so the legend prints a real beta, e.g. "original-small ($\beta$=1000)"
    arms = [(f"original-small ($\\beta$={best_beta})",
             ORIGSMALL_KEYS[BETAS.index(best_beta)], "tab:blue"),
            ("C2", KEY_C2, "tab:orange"), ("N1", KEY_N1, "tab:green")]
    fig, ax = plt.subplots(figsize=(6, 4))
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
    ax.set_title("Train run 5 — reward over training (mean $\\pm$ SE)")
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
    """The set of original-small keys the controller pruned (from the newest decisions ledger)."""
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
    best_beta = write_reward_table(run5, olds)
    write_beta_sweep_table(run5, load_pruned_keys())
    write_substitution(run5, olds)
    try:
        write_reward_curve(run5, best_beta)
    except Exception as e:  # a plotting failure must not lose the tables
        print(f"reward_curve skipped: {e}")
    print(f"wrote reward_table.tex, beta_sweep_table.tex, substitution_table.tex, "
          f"substitution.md, reward_curve.pdf into {PLOTS}")


if __name__ == "__main__":
    main()
