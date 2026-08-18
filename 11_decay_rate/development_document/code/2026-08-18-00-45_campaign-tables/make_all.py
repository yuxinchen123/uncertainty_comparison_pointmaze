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
    fig_champion_curves()
    fig_nonuniform_scatter()
    fig_adagrad_diagnostics()
