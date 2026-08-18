#!/usr/bin/env python
"""Generator for the 11_decay_rate development document: the ledger table (spliced between
the AUTO-GENERATED markers) and the three campaign figures.

Everything is read from the campaign's own artifacts — results.tsv for the ledger, the
experiment folders' records for the curves — through the fixed harness modules, so the
document cannot drift from the measured records.

Run:  /p/rlprojects/RND/.venvs/exploration/bin/python make_all.py
"""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DOCDIR = os.path.dirname(os.path.dirname(HERE))
PROJ = os.path.dirname(DOCDIR)
sys.path.insert(0, os.path.join(PROJ, "code"))
from decay_harness.metrics import target_curve  # noqa: E402

TEX = os.path.join(DOCDIR, "11_decay_rate_development_document.tex")
CAMPAIGN = os.path.join(PROJ, "experiments",
                        "2026-08-17-22-05_autoresearch_uniform-fullbatch_"
                        "cellmid108-center100_4096step_seed0-9")
FIGDIR = os.path.join(DOCDIR, "figures")


def inject_table(name: str, body: str) -> None:
    """Splice `body` between the AUTO-GENERATED markers for `name`; hard-fail if the marker
    pair is missing or duplicated."""
    start = f"% >>> AUTO-GENERATED TABLE START: {name}"
    end = f"% >>> AUTO-GENERATED TABLE END: {name}"
    tex = open(TEX).read()
    if tex.count(start) != 1 or tex.count(end) != 1:
        raise RuntimeError(f"marker pair for {name!r} missing or duplicated in {TEX}")
    head, rest = tex.split(start)
    _, tail = rest.split(end)
    open(TEX, "w").write(head + start + "\n" + body + "\n" + end + tail)
    print(f"spliced table {name!r}")


def load_ledger() -> list:
    """Parse results.tsv into row dicts (tab-separated, one experiment per row)."""
    rows = []
    with open(os.path.join(PROJ, "results.tsv")) as fh:
        header = fh.readline().rstrip("\n").split("\t")
        for line in fh:
            rows.append(dict(zip(header, line.rstrip("\n").split("\t"))))
    return rows


def ledger_table() -> str:
    """The campaign ledger as a longtable: one row per iteration experiment, numbers from
    results.tsv (which mirror each experiment's metrics.json)."""
    rows = load_ledger()
    # description holds "exp NNN <text>"; keep the number and a trimmed text
    out = [r"{\footnotesize",
           r"\begin{longtable}{@{}r >{\raggedright\arraybackslash}p{5.2cm} r r r r r@{}}",
           r"\caption{The campaign ledger: every iteration experiment, in submission order."
           r" Columns are the primary and secondary deviation metrics, the start deviation,"
           r" the per-position slope mean and standard deviation"
           r" (\Cref{sec:metrics}); (discarded) marks abandoned directions."
           r" Numbers are the experiment's own 10-seed metrics; the exact method per row is"
           r" the method file at the ledger's commit.}\label{tab:ledger}\\",
           r"\toprule",
           r"exp & method / finding & dev\_worst & dev\_mean & start & slope & sl.\ std \\",
           r"\midrule\endfirsthead",
           r"\toprule",
           r"exp & method / finding & dev\_worst & dev\_mean & start & slope & sl.\ std \\",
           r"\midrule\endhead"]
    for r in rows:
        desc = r["description"]
        num = desc.split()[1] if desc.startswith("exp ") else "?"
        # method slug = text between "exp NNN " and the first ":" (fall back to 40 chars)
        text = desc.split(" ", 2)[2] if desc.startswith("exp ") else desc
        slug = text.split(":")[0]
        slug = (slug[:58] + "\\,\\ldots") if len(slug) > 60 else slug
        mark = "" if r["status"] == "keep" else " (discarded)"
        out.append(
            f"{num} & {latex_escape(slug)}{mark} & {float(r['dev_worst']):.3f} & "
            f"{float(r['dev_mean']):.3f} & {float(r['start_dev']):.2f} & "
            f"{float(r['slope_mean']):.3f} & {float(r['slope_std']):.3f} \\\\")
    out += [r"\bottomrule", r"\end{longtable}}"]
    return "\n".join(out)




VAL_LABELS = {  # folder slug -> table row label (regime is the block, so it is not repeated)
    "val_101_adam_baseline_uniform30": ("uniform", "Adam $10^{-3}$, raw readout (control)"),
    "val_102_sgd1t_baseline_uniform30": ("uniform", "SGD-$1/t$ (parent project's best), raw readout"),
    "val_103_adagrad3e-3_uniform30": ("uniform", "AdaGrad $3{\\times}10^{-3}$ + initial-copy readout"),
    "val_105_shrink_relu_uniform30": ("uniform", "residual-encoded shrink (exact solve)"),
    "val_107_coinflip_adaptive_uniform30": ("uniform", "coin flips, adaptive dictionary"),
    "val_112_elliptical_sigma035_uniform30": ("uniform", "elliptical posterior readout ($\\sigma \\le 0.35$)"),
    "val_114_hadamardperm_uniform30": ("uniform", "Hadamard coins (column-permuted), adaptive dictionary"),
    "val_117_hadamardnocollide_uniform30": ("uniform", "Hadamard coins (collision-free spikes), adaptive dictionary"),
    "val_104_adagrad3e-3_nonuniform30": ("nonuniform", "AdaGrad $3{\\times}10^{-3}$ + initial-copy readout"),
    "val_106_shrink_relu_nonuniform30": ("nonuniform", "residual-encoded shrink (exact solve)"),
    "val_108_coinflip_adaptive_nonuniform30": ("nonuniform", "coin flips, adaptive dictionary"),
    "val_113_elliptical_sigma035_nonuniform30": ("nonuniform", "elliptical posterior readout ($\\sigma \\le 0.35$)"),
    "val_115_hadamardperm_nonuniform30": ("nonuniform", "Hadamard coins (column-permuted), adaptive dictionary"),
    "val_111_coinflip_hadamard_dense30": ("dense grid", "Hadamard coins, sign-only scramble (broken variant, kept as record)"),
    "val_116_hadamardperm_dense30": ("dense grid", "Hadamard coins (column-permuted), adaptive dictionary"),
    "val_118_hadamardnocollide_dense30": ("dense grid", "Hadamard coins (collision-free spikes), adaptive dictionary"),
}


def _mark(vals, fmt):
    """Bold the best (smallest) and underline the second-best in a column; ties all marked."""
    # before: vals = [0.05, 1.62, 0.05, 0.11]; after: two bolds (tied best), one underline
    order = sorted(set(vals))
    out = []
    for v in vals:
        t = fmt % v
        if v == order[0]:
            t = r"\textbf{" + t + "}"
        elif len(order) > 1 and v == order[1]:
            t = r"\underline{" + t + "}"
        out.append(t)
    return out


def validation_table() -> str:
    """The 30-seed validation table: regime blocks, methods sorted by dev_worst ascending,
    best bold / second-best underlined per deviation column within a block."""
    rows_by_regime = {}
    for slug, (regime, label) in VAL_LABELS.items():
        path = os.path.join(CAMPAIGN, slug, "metrics.json")
        if not os.path.exists(path):
            print(f"validation_table: MISSING {slug} (skipped)")
            continue
        with open(path) as fh:
            m = json.load(fh)
        rows_by_regime.setdefault(regime, []).append((label, m))
    out = [r"\begin{table}[H]", r"\centering", r"\footnotesize",
           r"\setlength{\tabcolsep}{4pt}",
           r"\begin{tabular}{@{}>{\raggedright\arraybackslash}p{5.6cm} r r r r r@{}}", r"\toprule",
           r"method & dev\_worst $\downarrow$ & dev\_mean & start\_dev & slope & slope std \\"]
    for regime in ("uniform", "nonuniform", "dense grid"):
        rows = rows_by_regime.get(regime, [])
        if not rows:
            continue
        rows.sort(key=lambda t: t[1]["dev_worst"])
        out.append(r"\midrule")
        out.append(r"\multicolumn{6}{@{}l}{\textbf{" + regime + r"} (30 seeds)} \\")
        dw = _mark([r[1]["dev_worst"] for r in rows], "%.3f")
        dm = _mark([r[1]["dev_mean"] for r in rows], "%.3f")
        for (label, m), a, b in zip(rows, dw, dm):
            out.append(f"{label} & {a} & {b} & {m['start_dev']:.2f} & "
                       f"{m.get('slope_mean', float('nan')):.3f} & "
                       f"{m.get('slope_std', float('nan')):.3f} \\\\")
    out += [r"\bottomrule", r"\end{tabular}",
            r"\caption{Validation at $30$ seeds. Uniform and nonuniform blocks pool the three"
            r" fixed point sets; the dense-grid block is the $432$-point capacity stress."
            r" Within each block rows are sorted by dev\_worst (the arrowed sort column;"
            r" lower is better), and the best and second-best dev\_worst and dev\_mean are"
            r" bold and underlined. start\_dev is $0$ for every method with a normalized"
            r" readout; the two raw-readout controls carry the untrained bonus spread."
            r" Slope columns are unmarked (closeness to $-1/2$, not size, is the goal).}",
            r"\label{tab:validation}", r"\end{table}"]
    return "\n".join(out)




NEURAL_ROWS = [  # (experiment folder, block, row label) for the phase-2 neural table
    ("exp_037_cfn_adam1e-4", "maze + AntMaze states (uniform)", "gradient coin-flip net, Adam $10^{-4}$ (paper rate)"),
    ("exp_039_cfn_adam1e-2", "maze + AntMaze states (uniform)", "gradient coin-flip net, Adam $10^{-2}$ (best gradient variant)"),
    ("exp_052_cfnreplay_vectors", "maze + AntMaze states (uniform)", "replay-buffer coin-flip net (4 Adam steps/visit)"),
    ("exp_048_deepshrink_vectors", "maze + AntMaze states (uniform)", "deep trunk + exact shrink head"),
    ("exp_049_deepcfn_vectors", "maze + AntMaze states (uniform)", "deep trunk + exact coin-flip head"),
    ("exp_044_cfnconv_adam1e-4_atari", "Atari frames (uniform)", "gradient coin-flip conv net, Adam $10^{-4}$"),
    ("exp_045_cfnconv_adam1e-3_atari", "Atari frames (uniform)", "gradient coin-flip conv net, Adam $10^{-3}$"),
    ("exp_046_cfnconv_adagrad1e-2_atari", "Atari frames (uniform)", "gradient coin-flip conv net, AdaGrad $10^{-2}$"),
    ("exp_050_deepshrink_atari", "Atari frames (uniform)", "conv trunk (256 feat.) + exact shrink head"),
    ("exp_051_deepcfn_atari", "Atari frames (uniform)", "conv trunk (256 feat.) + exact coin-flip head"),
    ("exp_059_deepshrink_wide_atari", "Atari frames (uniform)", "conv trunk (1024 feat.) + exact shrink head"),
    ("exp_060_deepcfn_wide_atari", "Atari frames (uniform)", "conv trunk (1024 feat.) + exact coin-flip head"),
    ("val_207_deepcfn_cappedwide_atari", "Atari frames (uniform)", "conv trunk (1024 feat.) + capped coin-flip head"),
    ("exp_053_deepshrink_heldout_vectors", "held-out 20\% (vectors)", "deep trunk + exact shrink head"),
    ("exp_054_deepcfn_heldout_vectors", "held-out 20\% (vectors)", "deep trunk + exact coin-flip head"),
    ("exp_057_deepshrink_heldout_atari", "held-out 20\% (Atari)", "conv trunk + exact shrink head"),
    ("exp_058_deepcfn_heldout_atari", "held-out 20\% (Atari)", "conv trunk + exact coin-flip head"),
    ("exp_056_deepshrink_nonuniform_vectors", "nonuniform visitation (vectors)", "deep trunk + exact shrink head"),
    ("exp_055_deepcfn_nonuniform_vectors", "nonuniform visitation (vectors)", "deep trunk + exact coin-flip head"),
    ("val_206_deepcfn_capped_nonuniform30", "nonuniform visitation (vectors)", "deep trunk + capped coin-flip head (30 seeds)"),
    ("val_209_deepcfn_capped_heldout30", "held-out 20\% (vectors)", "deep trunk + capped coin-flip head (30 seeds)"),
    ("val_208_deepcfn_cappedwide_atari_nonuniform", "nonuniform visitation (Atari)", "conv trunk (1024 feat.) + capped coin-flip head"),
]


def neural_table() -> str:
    """The phase-2 neural-ladder table: blocks by domain/regime, rows sorted by dev_worst."""
    blocks = {}
    for exp, block, label in NEURAL_ROWS:
        path = os.path.join(CAMPAIGN, exp, "metrics.json")
        if not os.path.exists(path):
            print(f"neural_table: MISSING {exp} (skipped)")
            continue
        with open(path) as fh:
            blocks.setdefault(block, []).append((label, json.load(fh)))
    out = [r"\begin{table}[H]", r"\centering", r"\footnotesize",
           r"\setlength{\tabcolsep}{4pt}",
           r"\begin{tabular}{@{}>{\raggedright\arraybackslash}p{6.4cm} r r r r@{}}",
           r"\toprule",
           r"method & dev\_worst $\downarrow$ & dev\_mean & slope & slope std \\"]
    order = ["maze + AntMaze states (uniform)", "Atari frames (uniform)",
             "held-out 20\% (vectors)", "held-out 20\% (Atari)",
             "nonuniform visitation (vectors)", "nonuniform visitation (Atari)"]
    for block in order:
        rows = blocks.get(block, [])
        if not rows:
            continue
        rows.sort(key=lambda t: t[1]["dev_worst"])
        out.append(r"\midrule")
        out.append(r"\multicolumn{5}{@{}l}{\textbf{" + block + r"} (10 seeds)} \\")
        dw = _mark([r[1]["dev_worst"] for r in rows], "%.3f")
        dm = _mark([r[1]["dev_mean"] for r in rows], "%.3f")
        for (label, m), a, b in zip(rows, dw, dm):
            out.append(f"{label} & {a} & {b} & "
                       f"{m.get('slope_mean', float('nan')):.3f} & "
                       f"{m.get('slope_std', float('nan')):.3f} \\\\")
    out += [r"\bottomrule", r"\end{tabular}",
            r"\caption{The phase-2 neural ladder. Blocks are domain--regime combinations;"
            r" within each block rows are sorted by dev\_worst (lower is better; best and"
            r" second-best dev\_worst and dev\_mean bold and underlined). Every method"
            r" starts at exactly 1 (start\_dev 0, omitted). In the held-out blocks the"
            r" metric scores trained states against their counts AND the never-trained"
            r" 20\% against the constant oracle value 1, so it punishes a bonus that"
            r" generalizes the decay onto states never actually visited.}",
            r"\label{tab:neural}", r"\end{table}"]
    return "\n".join(out)




def fig_neural_curves() -> None:
    """Phase-2 story in three panels: gradient training lags (vectors), the 256-feature
    capacity floor (Atari), and the widened exact head (Atari)."""
    panels = [
        ("exp_039_cfn_adam1e-2", "cell_midpoints", "gradient coin-flip net (best variant)"),
        ("exp_051_deepcfn_atari", "atari_frames", "coin-flip head, 256 features (floor)"),
        ("exp_060_deepcfn_wide_atari", "atari_frames", "coin-flip head, 1024 features"),
    ]
    plt.rcParams.update({"font.size": 13})
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), sharey=True)
    for ax, (exp, env, title) in zip(axes, panels):
        steps, b, m = seed_mean(load_records(exp, env))
        w = steps >= 1
        for i in range(b.shape[1]):
            ax.loglog(steps[w], np.maximum(b[w, i], 1e-4), color="tab:blue", alpha=0.12, lw=0.4)
        ax.loglog(steps[w], np.minimum(1.0, steps[w] ** -0.5), color="grey", lw=2)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("step $n$")
    axes[0].set_ylabel("bonus $b_i(n)$")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "neural_curves.pdf"))
    plt.close(fig)
    print("wrote neural_curves.pdf")


def latex_escape(s: str) -> str:
    """Escape the characters that appear in ledger descriptions."""
    return (s.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")
            .replace("#", r"\#").replace("->", r"$\to$").replace("+-", r"$\pm$"))


def load_records(exp: str, env: str = "cell_midpoints") -> list:
    """Load one experiment's non-diverged records for one point set."""
    recs = []
    for p in sorted(glob.glob(os.path.join(CAMPAIGN, exp, f"{env}_seed*.json"))):
        with open(p) as fh:
            r = json.load(fh)
        if not r.get("diverged"):
            recs.append(r)
    if not recs:
        raise RuntimeError(f"no records for {exp}/{env}")
    return recs


def seed_mean(recs: list) -> tuple:
    """(steps, seed-mean bonus (T, P), seed-mean counts (T, P))."""
    steps = np.asarray(recs[0]["checkpoint_steps"])
    b = np.mean([np.asarray(r["bonus"], dtype=float) for r in recs], axis=0)
    m = np.mean([np.asarray(r["visit_counts"], dtype=float) for r in recs], axis=0)
    return steps, b, m


def fig_champion_curves() -> None:
    """2x4-panel-in-a-row figure: per-position curve fans for four representative methods."""
    panels = [("exp_001_adam_baseline", "Adam (raw readout)"),
              ("exp_009_adagrad_lr3e-3", "AdaGrad $3{\\times}10^{-3}$ + init-copy"),
              ("exp_010_linhead_rls_residual_count", "residual-encoded shrink"),
              ("exp_019_coinflip_adaptive_uniform", "coin flips, adaptive dictionary")]
    plt.rcParams.update({"font.size": 13})
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8), sharey=True)
    for ax, (exp, title) in zip(axes, panels):
        steps, b, m = seed_mean(load_records(exp))
        w = steps >= 1
        for i in range(b.shape[1]):
            ax.loglog(steps[w], b[w, i], color="tab:blue", alpha=0.25, lw=0.5)
        ax.loglog(steps[w], np.minimum(1.0, steps[w] ** -0.5), color="grey", lw=2)
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("step $n$")
    axes[0].set_ylabel("bonus $b_i(n)$")
    fig.tight_layout()
    os.makedirs(FIGDIR, exist_ok=True)
    fig.savefig(os.path.join(FIGDIR, "champion_curves.pdf"))
    plt.close(fig)
    print("wrote champion_curves.pdf")


def fig_nonuniform_scatter() -> None:
    """Final bonus against final realized count per position: shrink vs coin flips."""
    panels = [("exp_011_linhead_rls_nonuniform", "residual-encoded shrink"),
              ("exp_020_coinflip_adaptive_nonuniform", "coin flips, adaptive dictionary")]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0), sharey=True)
    for ax, (exp, title) in zip(axes, panels):
        steps, b, m = seed_mean(load_records(exp))
        mf = np.maximum(m[-1], 1.0)
        ax.loglog(mf, b[-1], "o", ms=4, alpha=0.7)
        grid = np.logspace(np.log10(mf.min()), np.log10(mf.max()), 50)
        ax.loglog(grid, grid ** -0.5, color="grey", lw=2)
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("final visit count $m_i$")
    axes[0].set_ylabel("final bonus $b_i$")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "nonuniform_scatter.pdf"))
    plt.close(fig)
    print("wrote nonuniform_scatter.pdf")


def fig_adagrad_diagnostics() -> None:
    """Re-render the exp-007 diagnostics as a PDF via the campaign diagnose module."""
    # reuse diagnose.py's plotting by importing its pieces (fixed tool, not re-implemented)
    import diagnose
    from decay_harness.fitting import fit_power_floor
    from decay_harness.points import point_set
    recs = load_records("exp_007_adagrad_initcopy_norm")
    steps, b, m = seed_mean(recs)
    from decay_harness.metrics import deviation_per_position
    dev = np.mean([deviation_per_position(r) for r in recs], axis=0)
    pts = point_set("cell_midpoints")
    slopes = np.array([fit_power_floor(steps, b[:, i])["slope"] for i in range(b.shape[1])])
    order = np.argsort(-dev)
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.2))
    diagnose.maze_heatmap(axes[0], dev, pts, "per-position deviation")
    diagnose.maze_heatmap(axes[1], slopes, pts, "per-position fitted slope")
    ax = axes[2]
    w = steps >= 1
    for k in order[:6]:
        ax.loglog(steps[w], b[w, k], lw=1.0)
    ax.loglog(steps[w], np.minimum(1.0, steps[w] ** -0.5), color="grey", lw=2)
    ax.set_title("six worst positions", fontsize=10)
    ax.set_xlabel("step $n$")
    ax = axes[3]
    ax.hist(slopes, bins=40)
    ax.axvline(-0.5, color="grey", lw=2)
    ax.set_title("per-position slopes", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "adagrad_diagnostics.pdf"))
    plt.close(fig)
    print("wrote adagrad_diagnostics.pdf")


if __name__ == "__main__":
    inject_table("ledger", ledger_table())
    inject_table("validation", validation_table())
    inject_table("neural", neural_table())
    fig_champion_curves()
    fig_nonuniform_scatter()
    fig_adagrad_diagnostics()
    fig_neural_curves()
