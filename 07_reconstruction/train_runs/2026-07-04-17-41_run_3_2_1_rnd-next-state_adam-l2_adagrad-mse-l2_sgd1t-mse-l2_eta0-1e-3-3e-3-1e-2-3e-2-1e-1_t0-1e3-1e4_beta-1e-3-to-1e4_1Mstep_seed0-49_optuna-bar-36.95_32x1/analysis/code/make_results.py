#!/usr/bin/env python3
"""Regenerate all run-3.2.1 / run-3.2.2 analysis artifacts from the raw per-run JSONs:

  1. plots/reward_table.tex       -- run-3.2.1 six-row best-per-(optimizer, readout) table
  2. plots/validation_table.tex   -- run-3.2.2 three-row fresh-seed follow-up table
  3. plots/line_reward_curve.{pdf,png} -- training-reward curves for the swept best configs + references

The numbers are COMPUTED from the data (never hardcoded), then asserted against the known ground-truth
values below to two decimals, so re-running stays truthful. Scoring convention lives in common.py:
R_i = last train_history row's train/mean_extrinsic_reward; a config pools its seeds (Rbar, SE, success =
fraction with R_i > 5.0); only completed=true runs; a curve point needs >= 10 seeds.

Run 3.2.1 SELECTED the best cell per (optimizer, readout) as the max over many small-n cells (160 SGD-1/t
cells), which biases the selected mean upward. Run 3.2.2 re-ran the two best SGD-1/t configs plus the
canonical Adam-mse RND benchmark on 100 FRESH seeds (100-199) to check whether the advantage survives.
"""
from __future__ import annotations

import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

# ---------------------------------------------------------------------------------------------------------
# Data locations (this file lives at <R321>/analysis/code/make_results.py).
# ---------------------------------------------------------------------------------------------------------
CODE_DIR = Path(__file__).resolve().parent
RUN_DIR = CODE_DIR.parent.parent                 # the run-3.2.1 folder
TRAIN_RUNS = RUN_DIR.parent                       # .../07_reconstruction/train_runs
PLOTS = RUN_DIR / "analysis" / "plots"

R321_LOCAL = RUN_DIR / "data" / "2026-07-04-18-21_optimizer-variants-bar" / "local"
R311_LOCAL = (TRAIN_RUNS / "2026-06-26-05-15_run_3_1_1_pytorch_4algo_betagrid_ridge1-1e-2_input-sa-nexts_"
              "1Mstep_100seed_trainrewardonly" / "data" / "2026-06-26-05-45_4algo-betaridge-input" / "local")
R322_LOCAL = (TRAIN_RUNS / "2026-07-07-23-10_run_3_2_2_validate_sgd1t-l2-eta1e-1-t1e3-b1__sgd1t-mse-eta1e-1-"
              "t1e3-b10__adam-mse-b100-rndbench__rnd-next-state_seed100-199_100seed_1Mstep_cpu-nolim-resv_32x1"
              / "data" / "2026-07-07-23-15_validate" / "local")
RUN2_LOCAL = TRAIN_RUNS / "2026-06-24-21-24_run_2_after_reorganization" / "data" / "local"
# The two visit-count oracle sweeps (gt_position_velocity, beta=1), re-measured on the SAME
# training-episode-reward metric and fresh seeds (200-499) as the run-3.2.2 validation: the 1/sqrt(n)
# decay (alpha=-0.5) and the 1/n decay (alpha=-1). Added as the oracle-ceiling rows of the follow-up
# table. These sweeps may still be filling; oracle_stats pools whatever completed runs exist.
ORACLE_DATA = R322_LOCAL.parent.parent   # .../run_3_2_2.../data
ORACLE_1SQRTN_LOCAL = ORACLE_DATA / "2026-07-09-00-15_oracle-gt-position-velocity_seed200-499" / "local"
ORACLE_1N_LOCAL = ORACLE_DATA / "2026-07-09-00-27_oracle-gt-position-velocity-1n_seed200-499" / "local"

# The canonical RND benchmark config (Adam optimizer, mse readout, beta=100), used as row 4 of the
# run-3.2.1 table (from run-3.1.1 data) and as C2 of the run-3.2.2 table (from run-3.2.2 data).
BENCH_KEY = "adam|mse|-|-|100"

# ---------------------------------------------------------------------------------------------------------
# Ground-truth values the computed tables must reproduce (see the task spec). Keyed by config id.
# tuple = (Rbar, SE, success, n). The generator asserts each computed cell matches to two decimals.
# ---------------------------------------------------------------------------------------------------------
GT_R321 = {
    "sgd1t|mse|0.1|1000|10": (47.64, 5.24, 0.77, 48),
    "sgd1t|l2|0.01|10000|1": (45.33, 5.50, 0.79, 47),
    "adagrad|mse|-|-|100":   (38.35, 4.75, 0.77, 47),
    "adam|mse|-|-|100":      (35.15, 3.80, 0.75, 52),   # run-3.1.1 reference row
    "adagrad|l2|-|-|1":      (34.11, 4.79, 0.71, 48),
    "adam|l2|-|-|1":         (26.15, 4.11, 0.55, 49),
}
GT_R322 = {                                             # final, all n=100 (2026-07-09)
    "sgd1t|mse|0.1|1000|10": (37.12, 3.17, 0.84, 100),  # C1
    "adam|mse|-|-|100":      (34.50, 3.13, 0.69, 100),  # C2 (benchmark)
    "sgd1t|l2|0.1|1000|1":   (33.17, 3.28, 0.69, 100),  # C0
}


# ---------------------------------------------------------------------------------------------------------
# Aggregation.
# ---------------------------------------------------------------------------------------------------------
def aggregate(records) -> dict:
    """Pool each configuration's seeds into stats.

    Before: a list of RunRecord, many per config.
    After:  {config_key: {"key","optimizer","readout","eta0","t0","beta","n","mean","se","succ"}}.
    """
    by_cfg = defaultdict(list)   # config_key -> list of final training rewards
    meta = {}                    # config_key -> one RunRecord (for the config's identity fields)
    for r in records:
        if r.final_train_reward is None:
            continue
        k = r.config_key()
        by_cfg[k].append(r.final_train_reward)
        meta[k] = r
    out = {}
    for k, vals in by_cfg.items():
        n = len(vals)
        se = statistics.stdev(vals) / math.sqrt(n) if n > 1 else 0.0
        rr = meta[k]
        out[k] = {"key": k, "optimizer": rr.optimizer, "readout": rr.readout, "eta0": rr.eta0,
                  "t0": rr.t0, "beta": rr.beta, "n": n, "mean": statistics.mean(vals), "se": se,
                  "succ": sum(1 for v in vals if v > C.SUCCESS_THRESHOLD) / n}
    return out


def best_config(agg: dict, optimizer: str, readout: str, min_seeds: int = C.MIN_SEEDS) -> dict:
    """Highest-mean eligible (n >= min_seeds) config for one (optimizer, readout) pair. Raises if none.
    This is exactly the run-3.2.1 selection rule (the max over cells that inflates the selected mean)."""
    cands = [v for v in agg.values()
             if v["optimizer"] == optimizer and v["readout"] == readout and v["n"] >= min_seeds]
    if not cands:
        raise ValueError(f"no eligible config for optimizer={optimizer} readout={readout} (>= {min_seeds} seeds)")
    return max(cands, key=lambda v: v["mean"])


# ---------------------------------------------------------------------------------------------------------
# LaTeX value formatting.
# ---------------------------------------------------------------------------------------------------------
def fmt_pow(x) -> str:
    """Format a positive schedule/coefficient value for a table cell: the human-scale powers {0.1, 1, 10}
    print plainly ($0.1$/$1$/$10$), other exact powers of ten as $10^{k}$ ($10^{-2}$, $10^2$, $10^3$,
    $10^4$), any non-power as $x:g$. Matches the run-3.2.1/3.2.2 table style."""
    if x is None or not C._finite(x) or x <= 0:
        return "---"
    k = math.log10(x)
    kk = round(k)
    if abs(k - kk) < 1e-9:                 # exact power of ten
        if -1 <= kk <= 1:                  # 0.1, 1, 10 -> plain
            return f"${x:g}$"
        return f"$10^{{{int(kk)}}}$"
    return f"${x:g}$"


def phi(z: float) -> float:
    """Standard-normal CDF Phi(z) = 0.5 (1 + erf(z / sqrt(2)))."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _check(agg: dict, gt: dict, label: str) -> None:
    """Assert every ground-truth config in gt is present in agg and matches Rbar/SE/succ to two decimals
    and n exactly. Raises AssertionError naming the first mismatch (keeps the generator honest)."""
    for key, (m, se, succ, n) in gt.items():
        assert key in agg, f"[{label}] missing config {key}"
        a = agg[key]
        for name, want, got in (("Rbar", m, a["mean"]), ("SE", se, a["se"]), ("succ", succ, a["succ"])):
            assert round(got, 2) == round(want, 2), f"[{label}] {key} {name}: got {got:.4f} want {want}"
        assert a["n"] == n, f"[{label}] {key} n: got {a['n']} want {n}"


# ---------------------------------------------------------------------------------------------------------
# Artifact 1: run-3.2.1 six-row table.
# ---------------------------------------------------------------------------------------------------------
OPT_METHOD = {"sgd1t": "O3 SGD-$1/t$", "adagrad": "O2 AdaGrad", "adam": "O1 Adam"}
READOUT_TEX = {"mse": r"$\mathcal{B}_{\mathrm{mse}}$", "l2": r"$\mathcal{B}_{\mathrm{l2}}$"}


def reward_table(agg321: dict, agg311: dict) -> list[dict]:
    """Assemble the six table rows: the five best-per-(optimizer, readout) run-3.2.1 configs plus the
    run-3.1.1 Adam-mse reference, sorted by Rbar descending. Returns the row dicts (for the curve)."""
    # the five (optimizer, readout) pairs actually swept in run 3.2.1
    rows = [best_config(agg321, opt, ro) for (opt, ro) in
            [("sgd1t", "mse"), ("sgd1t", "l2"), ("adagrad", "mse"), ("adagrad", "l2"), ("adam", "l2")]]
    for r in rows:
        r["is_ref"] = False
    # row 4: the canonical Adam-mse RND benchmark, taken from run-3.1.1 data (not swept in run 3.2.1)
    ref = dict(agg311[BENCH_KEY]); ref["is_ref"] = True
    rows.append(ref)
    rows.sort(key=lambda r: r["mean"], reverse=True)
    return rows


def write_reward_table(rows: list[dict]) -> None:
    """Write plots/reward_table.tex: a bare tabular. Bold best / underline second in the Rbar and succ
    columns only (marks computed on the rounded display values so display-precision ties mark together)."""
    # bold/underline on rounded display values so the tied 0.77 success pair both underline
    rbar_best, rbar_second = C.rank_marks([round(r["mean"], 2) for r in rows], lower_is_better=False)
    succ_best, succ_second = C.rank_marks([round(r["succ"], 2) for r in rows], lower_is_better=False)
    lines = []
    for i, r in enumerate(rows):
        method = OPT_METHOD[r["optimizer"]] + (" (run-3.1.1 ref.)" if r["is_ref"] else "")
        rbar = f"{r['mean']:.2f}"
        if i in rbar_best:
            rbar = f"\\textbf{{{rbar}}}"
        elif i in rbar_second:
            rbar = f"\\underline{{{rbar}}}"
        succ = f"{r['succ']:.2f}"
        if i in succ_best:
            succ = f"\\textbf{{{succ}}}"
        elif i in succ_second:
            succ = f"\\underline{{{succ}}}"
        eta = fmt_pow(r["eta0"]) if r["optimizer"] == "sgd1t" else "---"
        t0 = fmt_pow(r["t0"]) if r["optimizer"] == "sgd1t" else "---"
        lines.append(
            f"{method} & {READOUT_TEX[r['readout']]} & {fmt_pow(r['beta'])} & {eta} & {t0} & "
            f"{rbar} & {r['se']:.2f} & {succ} & {int(r['n'])} \\\\"
        )
    body = "\n".join(lines)
    (PLOTS / "reward_table.tex").write_text(
        "% Run 3.2.1: best configuration per (optimizer, readout), scored on final training-episode\n"
        "% extrinsic reward. Row 4 (Adam, mse) is the run-3.1.1 reference arm (not swept in run 3.2.1).\n"
        "% Higher reward is better; rows sorted by Rbar descending; in the Rbar and succ. columns the best\n"
        "% is bold and the second-best underlined.\n"
        "\\begin{tabular}{@{}>{\\raggedright\\arraybackslash}p{2.5cm} c c c c r r r r@{}}\n"
        "\\toprule\n"
        "\\textbf{Method} & \\textbf{Readout} & \\textbf{$\\beta^\\star$} & \\textbf{$\\eta_0^\\star$} & "
        "\\textbf{$t_0^\\star$} & \\textbf{$\\bar{R}$ ($\\uparrow$)} & \\textbf{SE} & \\textbf{succ.} & "
        "\\textbf{$n$} \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
    )


# ---------------------------------------------------------------------------------------------------------
# Artifact 2: run-3.2.2 three-row validation table.
# ---------------------------------------------------------------------------------------------------------
OPT_NAME = {"sgd1t": "SGD-$1/t$", "adagrad": "AdaGrad", "adam": "Adam"}
CONFIG_LABEL = {"sgd1t|mse|0.1|1000|10": "C1", "adam|mse|-|-|100": "C2 (RND benchmark)",
                "sgd1t|l2|0.1|1000|1": "C0",
                "oracle|1sqrtn": "O$\\sqrt{}$ (oracle)", "oracle|1n": "O$n$ (oracle)"}
# per-oracle "Optimizer / readout / beta" descriptor + row label for the validation table
ORACLE_SPECS = [
    (ORACLE_1SQRTN_LOCAL, "oracle|1sqrtn", "visit-count / $1/\\sqrt{n}$ / $1$"),
    (ORACLE_1N_LOCAL,     "oracle|1n",     "visit-count / $1/n$ / $1$"),
]


def oracle_stats(local_dir, key: str, desc: str) -> dict | None:
    """Pool one gt_position_velocity oracle sweep's COMPLETED runs into a validation-table row dict.
    Returns None if the sweep dir is missing/empty, so the table still builds before the sweep finishes.

    Before: a dir of per-run JSONs, each with train_history[-1]['train/mean_extrinsic_reward'].
    After:  {"key","desc","n","mean","se","succ"}  (same fields the table's row loop reads, minus optimizer)."""
    import glob
    import json
    vals = []
    for f in glob.glob(str(local_dir / "*.json")):
        try:
            r = json.load(open(f))
        except Exception:
            continue
        if not r.get("completed", True):
            continue
        th = r.get("train_history") or []
        if th:
            vals.append(th[-1]["train/mean_extrinsic_reward"])
    n = len(vals)
    if n == 0:
        return None
    se = statistics.stdev(vals) / math.sqrt(n) if n > 1 else 0.0
    return {"key": key, "desc": desc, "n": n, "mean": statistics.mean(vals), "se": se,
            "succ": sum(1 for v in vals if v > C.SUCCESS_THRESHOLD) / n}


def write_validation_table(agg322: dict) -> None:
    """Write plots/validation_table.tex: the three fresh-seed RND configs PLUS the two visit-count oracle
    rows (1/sqrt(n) and 1/n, gt_position_velocity beta=1 on the same fresh seeds), sorted by Rbar
    descending, with the gap to the RND benchmark and a one-sided normal confidence that the row beats
    the benchmark: Delta = Rbar_row - Rbar_bench, conf = Phi(Delta / sqrt(SE_row^2 + SE_bench^2))."""
    bench = agg322[BENCH_KEY]
    # merge the three RND configs with whatever oracle rows have data yet (skip an unfinished sweep)
    oracle_rows = [row for spec in ORACLE_SPECS if (row := oracle_stats(*spec)) is not None]
    rows = sorted(list(agg322.values()) + oracle_rows, key=lambda r: r["mean"], reverse=True)
    rbar_best, rbar_second = C.rank_marks([round(r["mean"], 2) for r in rows], lower_is_better=False)
    succ_best, succ_second = C.rank_marks([round(r["succ"], 2) for r in rows], lower_is_better=False)
    lines = []
    for i, r in enumerate(rows):
        rbar = f"{r['mean']:.2f}"
        if i in rbar_best:
            rbar = f"\\textbf{{{rbar}}}"
        elif i in rbar_second:
            rbar = f"\\underline{{{rbar}}}"
        succ = f"{r['succ']:.2f}"
        if i in succ_best:
            succ = f"\\textbf{{{succ}}}"
        elif i in succ_second:
            succ = f"\\underline{{{succ}}}"
        # oracle rows carry a ready-made descriptor; RND config rows build it from optimizer/readout/beta
        cfg_desc = r["desc"] if "desc" in r else f"{OPT_NAME[r['optimizer']]} / {r['readout']} / {fmt_pow(r['beta'])}"
        if r["key"] == BENCH_KEY:
            delta_s, conf_s = "--- (ref.)", "---"
        else:
            # gap and one-sided normal confidence from the RAW (full-precision) means and SEs
            d = r["mean"] - bench["mean"]
            z = d / math.sqrt(r["se"] ** 2 + bench["se"] ** 2)
            delta_s = f"${d:+.2f}$"
            conf_s = f"{round(100 * phi(z))}\\%"
        lines.append(
            f"{CONFIG_LABEL[r['key']]} & {cfg_desc} & {rbar} & {r['se']:.2f} & {succ} & {int(r['n'])} & "
            f"{delta_s} & {conf_s} \\\\"
        )
    body = "\n".join(lines)
    (PLOTS / "validation_table.tex").write_text(
        "% Run 3.2.2 follow-up: the two best run-3.2.1 SGD-1/t configs, the Adam-mse RND benchmark (all on\n"
        "% fresh seeds 100-199), and the two visit-count oracle rows (gt_position_velocity beta=1 on fresh\n"
        "% seeds 200-499: 1/sqrt(n) decay and 1/n decay). Rows sorted by Rbar descending; best bold / second\n"
        "% underlined in the Rbar and succ. columns. Delta = row Rbar - benchmark Rbar; conf. = one-sided\n"
        "% normal confidence Phi(Delta / sqrt(SE_row^2 + SE_bench^2)) that the row exceeds the benchmark.\n"
        "\\begin{tabular}{@{}l l r r r r r r@{}}\n"
        "\\toprule\n"
        "\\textbf{Config} & \\textbf{Optimizer / readout / $\\beta$} & \\textbf{$\\bar{R}$ ($\\uparrow$)} & "
        "\\textbf{SE} & \\textbf{succ.} & \\textbf{$n$} & \\textbf{$\\Delta$ vs bench.} & \\textbf{conf.} \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
    )


# ---------------------------------------------------------------------------------------------------------
# Artifact 3: training-reward curves.
# ---------------------------------------------------------------------------------------------------------
def pooled_curve(curves: list[list[tuple[int, float]]], min_seeds: int):
    """Pool per-run (step, reward) curves into per-step mean / SE over seeds, keeping only steps reached by
    at least min_seeds runs.

    Before: [[(50000, 0.0), (1e6, 47.6)], [(50000, 0.1), ...], ...]  (one list per seed).
    After:  (steps, means, ses) each a list over the retained steps, ascending.
    """
    by_step = defaultdict(list)   # step -> list of that step's per-seed values
    for c in curves:
        for step, val in c:
            by_step[step].append(val)
    steps = sorted(s for s, v in by_step.items() if len(v) >= min_seeds)
    means = [statistics.mean(by_step[s]) for s in steps]
    ses = [statistics.stdev(by_step[s]) / math.sqrt(len(by_step[s])) if len(by_step[s]) > 1 else 0.0
           for s in steps]
    return steps, means, ses


# the five swept best configs (solid lines), in a fixed color order; labels use plain text (no LaTeX)
SOLID_SERIES = [
    ("sgd1t|mse|0.1|1000|10", "sgd1t mse (b=10)"),
    ("sgd1t|l2|0.01|10000|1", "sgd1t l2 (b=1)"),
    ("adagrad|mse|-|-|100",   "adagrad mse (b=100)"),
    ("adagrad|l2|-|-|1",      "adagrad l2 (b=1)"),
    ("adam|l2|-|-|1",         "adam l2 (b=1)"),
]


def curve_figure(records321, records311, records2) -> bool:
    """Draw the training-reward curves: five solid lines (run-3.2.1 best configs, training reward), one
    black dashed reference (run-3.1.1 Adam-mse benchmark, training reward), and one gray dashed run-2
    oracle (gt_position_velocity, EVAL reward). Each line is truncated to the largest step reached by
    >= 10 seeds. Returns True if the run-2 oracle line was drawn."""
    # index run-3.2.1 records by config key for the five solid series
    by_key321 = defaultdict(list)
    for r in records321:
        by_key321[r.config_key()].append(r)

    cmap = matplotlib.colormaps["tab10"]
    fig, ax = plt.subplots(figsize=(8.6, 5.6), dpi=150)

    # five solid lines: run-3.2.1 best configs, training-episode extrinsic reward, +-1 SE band
    for k, (key, label) in enumerate(SOLID_SERIES):
        steps, means, ses = pooled_curve([r.train_curve for r in by_key321[key]], C.MIN_SEEDS)
        color = cmap(k)
        ax.plot(steps, means, color=color, lw=1.9, ls="-", label=label)
        ax.fill_between(steps, [m - s for m, s in zip(means, ses)], [m + s for m, s in zip(means, ses)],
                        color=color, alpha=0.16, linewidth=0)

    # black dashed reference: run-3.1.1 Adam-mse RND benchmark, training reward, +-1 SE band
    bench = [r for r in records311 if r.config_key() == BENCH_KEY]
    steps, means, ses = pooled_curve([r.train_curve for r in bench], C.MIN_SEEDS)
    ax.plot(steps, means, color="black", lw=1.9, ls="--",
            label="adam mse (b=100) [RND benchmark, run 3.1.1]")
    ax.fill_between(steps, [m - s for m, s in zip(means, ses)], [m + s for m, s in zip(means, ses)],
                    color="black", alpha=0.10, linewidth=0)

    # gray dashed run-2 oracle: gt_position_velocity (beta=1), EVAL reward (a different metric -- noted)
    oracle = [r for r in records2 if r.algorithm == "gt_position_velocity" and abs(r.beta - 1.0) < 1e-9]
    drew_oracle = False
    steps, means, ses = pooled_curve([r.eval_curve for r in oracle], C.MIN_SEEDS)
    if steps:
        ax.plot(steps, means, color="0.45", lw=1.9, ls="--",
                label="gt_position_velocity (b=1) [oracle, run-2 eval]")
        ax.fill_between(steps, [m - s for m, s in zip(means, ses)], [m + s for m, s in zip(means, ses)],
                        color="0.45", alpha=0.12, linewidth=0)
        drew_oracle = True

    ax.set_xlabel("environment step")
    ax.set_ylabel("training-episode extrinsic reward")
    ax.set_title("Run 3.2.1 best optimizer configs: training-episode reward vs step "
                 "(mean $\\pm$ 1 SE over seeds, $n\\geq10$)")
    ax.grid(linestyle=":", alpha=0.5)
    ax.legend(fontsize=8.5, loc="upper left", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(PLOTS / "line_reward_curve.pdf", bbox_inches="tight")
    fig.savefig(PLOTS / "line_reward_curve.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    return drew_oracle


# ---------------------------------------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------------------------------------
def main() -> None:
    """Load all three runs, verify against ground truth, and write the two tables + the curve figure."""
    PLOTS.mkdir(parents=True, exist_ok=True)

    # load: run-3.2.1 (slow, ~9000 files), run-3.1.1 (only rnd_next_state), run-3.2.2, run-2 (oracle)
    print("[make_results] loading run-3.2.1 (slow)...", file=sys.stderr, flush=True)
    rec321 = C.load_records(R321_LOCAL)
    rec311 = C.load_records(R311_LOCAL, algorithm_filter="rnd_next_state")
    rec322 = C.load_records(R322_LOCAL)
    rec2 = C.load_records(RUN2_LOCAL, algorithm_filter="gt_position_velocity")
    print(f"[make_results] loaded: 3.2.1={len(rec321)} 3.1.1(rnd_next_state)={len(rec311)} "
          f"3.2.2={len(rec322)} run-2(gt)={len(rec2)}", file=sys.stderr, flush=True)

    agg321, agg311, agg322 = aggregate(rec321), aggregate(rec311), aggregate(rec322)

    # build + verify the six-row run-3.2.1 table
    rows = reward_table(agg321, agg311)
    _check({r["key"]: r for r in rows}, GT_R321, "run-3.2.1")
    write_reward_table(rows)

    # build + verify the three-row run-3.2.2 table
    _check(agg322, GT_R322, "run-3.2.2")
    write_validation_table(agg322)

    # curve figure
    drew_oracle = curve_figure(rec321, rec311, rec2)

    # report + the extra diagnostics used in analysis.md (each R322 config's own run-3.2.1 value)
    print("\n[make_results] run-3.2.1 table (sorted by Rbar):", file=sys.stderr)
    for r in rows:
        tag = " [run-3.1.1 ref]" if r["is_ref"] else ""
        print(f"  {r['key']:26s} Rbar={r['mean']:6.2f} SE={r['se']:.2f} succ={r['succ']:.2f} "
              f"n={r['n']}{tag}", file=sys.stderr)
    print("\n[make_results] run-3.2.2 validation vs its own run-3.2.1 selected value:", file=sys.stderr)
    for key in ("sgd1t|mse|0.1|1000|10", "sgd1t|l2|0.1|1000|1", "adam|mse|-|-|100"):
        v322 = agg322.get(key)
        v321 = agg321.get(key)
        v321s = f"{v321['mean']:.2f} (n={v321['n']})" if v321 else "n/a"
        print(f"  {key:26s} 3.2.2 Rbar={v322['mean']:.2f} (n={v322['n']})  <-  3.2.1 Rbar={v321s}",
              file=sys.stderr)
    # the run-3.2.1 best sgd1t|l2 config (row 2 of the table) for context vs C0
    row2 = best_config(agg321, "sgd1t", "l2")
    print(f"  best sgd1t|l2 in 3.2.1 (table row 2): {row2['key']} Rbar={row2['mean']:.2f} (n={row2['n']})",
          file=sys.stderr)
    print(f"\n[make_results] wrote reward_table.tex, validation_table.tex, "
          f"line_reward_curve.pdf/png (run-2 oracle drawn: {drew_oracle})", file=sys.stderr)


if __name__ == "__main__":
    main()
