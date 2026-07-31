#!/usr/bin/env python
"""Generate the run-3.2.3 results table and the run-3.2.4 validation table for main.tex.

Outputs (each a bare tabular, \\input by main.tex):
- ../plots/reward_table.tex      best configuration per (arm x bias scheme) in run 3.2.3, plus the
                                 run-3.2.2 RND-benchmark reference row (recomputed from data).
- ../plots/validation_table.tex  the top-5 configs (rank 1..rank 5) on fresh seeds 500-599 (run 3.2.4)
                                 vs the RND benchmark, with the gap and one-sided confidence.

Both tables score the final training-episode extrinsic reward (last train_history row's
train/mean_extrinsic_reward), completed records only. Marking follows the analysis convention:
rows sorted by Rbar descending, best bold / second underlined in the Rbar and succ. columns.
Snapshot tables: rerun this script when the sweeps complete.

Run:  /p/rlprojects/RND/.venvs/exploration/bin/python make_results.py
"""
import glob
import json
import math
import os
import re
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RUN3 = os.path.dirname(os.path.dirname(HERE))
TRAIN_RUNS = os.path.dirname(RUN3)
RUN4 = glob.glob(os.path.join(TRAIN_RUNS, "*run_3_2_4*"))[0]
RUN2_LOCAL = os.path.join(
    TRAIN_RUNS,
    "2026-07-07-23-10_run_3_2_2_validate_sgd1t-l2-eta1e-1-t1e3-b1__sgd1t-mse-eta1e-1-t1e3-b10__"
    "adam-mse-b100-rndbench__rnd-next-state_seed100-199_100seed_1Mstep_cpu-nolim-resv_32x1",
    "data", "2026-07-07-23-15_validate", "local")
PLOTS = os.path.join(os.path.dirname(HERE), "plots")
SUCCESS_THRESHOLD = 5.0
Z = math.sqrt(2.0)

# fast text extraction: the records carry a large train_episode_history, so full json.load over
# ~5k files is slow; every needed field sits in the small config/train_history parts -> regex.
RE_COMPLETED = re.compile(r'"completed":\s*(true|false)')
RE_MEAN = re.compile(r'"train/mean_extrinsic_reward":\s*([-\d.eE+]+)')


def _g(pat, txt):
    """First regex group in txt, or None."""
    m = re.search(pat, txt)
    return m.group(1) if m else None


def record_fields(txt):
    """(config_key, final_reward) from one record's raw text; (None, None) if unusable.
    before: raw JSON text; after: ('sgd1t|l2|0.1|10000|1|normal_0.5|-', 41.7)."""
    c = RE_COMPLETED.search(txt)
    if not c or c.group(1) != "true":
        return None, None
    means = RE_MEAN.findall(txt)
    if not means:
        return None, None
    opt = _g(r'"rnd_optimizer":\s*"([^"]+)"', txt)
    if opt is None:
        return None, None
    ro = _g(r'"rnd_bonus_readout":\s*"([^"]+)"', txt)
    beta = _g(r'"beta":\s*([-\d.eE+]+)', txt)
    bias = _g(r'"rnd_bias_init":\s*"([^"]+)"', txt) or "zero"
    norm = _g(r'"rnd_reward_norm":\s*(true|false)', txt) or "false"
    if opt == "sgd1t":
        eta0 = "%g" % float(_g(r'"rnd_sgd_eta0":\s*([-\d.eE+]+)', txt))
        t0 = "%g" % float(_g(r'"rnd_sgd_t0":\s*([-\d.eE+]+)', txt))
    else:
        eta0, t0 = "-", "-"
    key = (f"{opt}|{ro}|{eta0}|{t0}|" + ("%g" % float(beta)) + f"|{bias}|"
           + ("rewardnorm" if norm == "true" else "-"))
    return key, float(means[-1])


def load_scores(local_glob):
    """{config_key: [final rewards]} over every completed record matched by the glob."""
    scores = {}
    for path in glob.glob(local_glob):
        try:
            txt = open(path).read()
        except OSError:
            continue
        key, val = record_fields(txt)
        if key is not None:
            scores.setdefault(key, []).append(val)
    return scores


def agg(vals):
    """(mean, SE, succ, n) with SE = s/sqrt(n) (ddof=1) and succ = fraction > threshold."""
    n = len(vals)
    mean = statistics.mean(vals)
    se = statistics.stdev(vals) / math.sqrt(n) if n > 1 else float("nan")
    succ = sum(1 for v in vals if v > SUCCESS_THRESHOLD) / n
    return mean, se, succ, n


def phi(z):
    """Standard-normal CDF."""
    return 0.5 * (1.0 + math.erf(z / Z))


def tex_pow(x):
    """TeX for a number, powers of ten as 10^{k}. before: 10000.0 -> after: '$10^{4}$';
    1 -> '$1$'; 0.1 -> '$10^{-1}$'; 3 -> '$3$'."""
    v = float(x)
    if v > 0:
        k = math.log10(v)
        if abs(k - round(k)) < 1e-9:
            k = int(round(k))
            return "$1$" if k == 0 else ("$10$" if k == 1 else f"$10^{{{k}}}$")
    return f"${v:g}$"


def mark(rows, idx):
    """Bold the best and underline the second-best value at column idx (higher is better).
    rows = list of lists of floats/strings; returns the formatted strings for that column."""
    vals = [(r[idx], i) for i, r in enumerate(rows)]
    order = sorted(vals, key=lambda t: -t[0])
    out = {}
    for rank, (v, i) in enumerate(order):
        s = f"{v:.2f}"
        if rank == 0 or (rank > 0 and v == order[0][0]):
            s = f"\\textbf{{{s}}}"
        elif rank == 1 or (rank > 1 and v == order[1][0]):
            s = f"\\underline{{{s}}}"
        out[i] = s
    return out


def c2_reference():
    """The run-3.2.2 RND-benchmark row recomputed from its data (adam/mse/beta=100)."""
    vals = []
    for path in glob.glob(os.path.join(RUN2_LOCAL, "*.json")):
        try:
            rec = json.load(open(path))
        except (json.JSONDecodeError, OSError):
            continue
        if (rec.get("algorithm") != "rnd_next_state" or not rec.get("completed", True)
                or rec.get("rnd_optimizer") != "adam" or rec.get("rnd_bonus_readout") != "mse"
                or float(rec.get("beta", 0)) != 100.0):
            continue
        rows = rec.get("train_history") or []
        if rows and rows[-1].get("train/mean_extrinsic_reward") is not None:
            vals.append(float(rows[-1]["train/mean_extrinsic_reward"]))
    return agg(vals)


ARMS = [  # (row label, bias cell, selector over config-key fields)
    ("Adam (reward-norm)", "zero",
     lambda f: f[6] == "rewardnorm"),
    ("Adam", r"\texttt{pytorch\_\allowbreak default}",
     lambda f: f[0] == "adam" and f[6] == "-" and f[5] == "pytorch_default"),
    ("Adam", r"\texttt{normal\_0.5}",
     lambda f: f[0] == "adam" and f[6] == "-" and f[5] == "normal_0.5"),
    ("SGD-$1/t$", r"\texttt{pytorch\_\allowbreak default}",
     lambda f: f[0] == "sgd1t" and f[5] == "pytorch_default"),
    ("SGD-$1/t$", r"\texttt{normal\_0.5}",
     lambda f: f[0] == "sgd1t" and f[5] == "normal_0.5"),
]


def make_reward_table(sc3):
    """Write reward_table.tex: best config (n>=10) per arm x bias scheme + the RND-benchmark reference."""
    rows = []
    for label, bias_cell, sel in ARMS:
        # best key of this arm by mean over completed seeds (n >= 10 so a lucky n=2 cannot win)
        cands = [(statistics.mean(v), k) for k, v in sc3.items()
                 if len(v) >= 10 and sel(k.split("|"))]
        if not cands:
            continue
        _, k = max(cands)
        f = k.split("|")
        mean, se, succ, n = agg(sc3[k])
        readout = rf"$\mathcal{{B}}_{{\mathrm{{{f[1]}}}}}$"
        eta0 = tex_pow(f[2]) if f[2] != "-" else "---"
        t0 = tex_pow(f[3]) if f[3] != "-" else "---"
        rows.append([label, bias_cell, readout, tex_pow(f[4]), eta0, t0, mean, se, succ, n])
    m, s, sc, n = c2_reference()
    rows.append(["run-3.2.2 benchmark", "zero", r"$\mathcal{B}_{\mathrm{mse}}$",
                 "$10^{2}$", "---", "---", m, s, sc, n])
    rows.sort(key=lambda r: -r[6])
    rbar = mark(rows, 6)
    succ_col = mark(rows, 8)
    lines = [
        "% Run 3.2.3: best configuration (>=10 seeds) per arm x bias scheme, scored on the final",
        "% training-episode extrinsic reward; the RND-benchmark row = the run-3.2.2 fresh-seed RND benchmark",
        "% (recomputed from its records). Rows sorted by Rbar descending; best bold / second",
        "% underlined in the Rbar and succ. columns. SNAPSHOT while the sweep tail runs —",
        "% regenerated by analysis/code/make_results.py.",
        r"\begin{tabular}{@{}>{\raggedright\arraybackslash}p{3.4cm} l c c c c r r r r@{}}",
        r"\toprule",
        r"\textbf{Method} & \textbf{bias init} & \textbf{Readout} & \textbf{$\beta^\star$} & "
        r"\textbf{$\eta_0^\star$} & \textbf{$t_0^\star$} & \textbf{$\bar{R}$ ($\uparrow$)} & "
        r"\textbf{SE} & \textbf{succ.} & \textbf{$n$} \\",
        r"\midrule",
    ]
    for i, r in enumerate(rows):
        lines.append(f"{r[0]} & {r[1]} & {r[2]} & {r[3]} & {r[4]} & {r[5]} & "
                     f"{rbar[i]} & {r[7]:.2f} & {succ_col[i]} & {r[9]} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(PLOTS, "reward_table.tex"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"reward_table.tex: {len(rows)} rows")


def make_validation_table(sc4, sc3):
    """Write validation_table.tex: rank 1..rank 5 fresh-seed rows vs the RND benchmark."""
    bench_m, bench_se, bench_succ, bench_n = c2_reference()
    # V-rank = the SELECTION order recorded in run 3.2.4's queue builder (rank field), fixed at
    # 2026-07-14 — the in-sample means keep moving, so ranking by them would relabel the rows
    sys.path.insert(0, os.path.join(RUN4, "slurm"))
    import build_queue as bq4
    rank_of_key = {}
    for c in bq4.CONFIGS:
        rank_of_key[bq4.config_key(c["params"], c["beta"])] = c["rank"]
    ranked = sorted(sc4, key=lambda k: rank_of_key.get(k, 99))
    rows = []
    for k in ranked:
        f = k.split("|")
        mean, se, succ, n = agg(sc4[k])
        opt = {"adam": "Adam", "sgd1t": "SGD-$1/t$"}[f[0]]
        method = f"{opt} / {f[1]} / " + tex_pow(f[4]).strip("$")
        bias = {"zero": "zero (+reward-norm)" if f[6] == "rewardnorm" else "zero",
                "pytorch_default": r"\texttt{pytorch\_\allowbreak default}",
                "normal_0.5": r"\texttt{normal\_0.5}"}[f[5]]
        rows.append([f"rank {rank_of_key.get(k, '?')}", f"{method}", bias, mean, se, succ, n])
    rows.append(["run-3.2.2 benchmark", "Adam / mse / 10^{2}", "zero",
                 bench_m, bench_se, bench_succ, bench_n])
    rows.sort(key=lambda r: -r[3])
    rbar = mark(rows, 3)
    succ_col = mark(rows, 5)
    lines = [
        "% Run 3.2.4: the run-3.2.3 top-5 (rank 1..rank 5 by in-sample mean at selection) on 100",
        "% FRESH seeds 500-599, no pruning, vs the run-3.2.2 RND benchmark (recomputed).",
        "% Delta = row Rbar - benchmark Rbar; conf. = one-sided normal confidence",
        "% Phi(Delta / sqrt(SE_row^2 + SE_bench^2)) that the row exceeds the benchmark.",
        "% SNAPSHOT while the validation run executes - regenerated by make_results.py.",
        r"\begin{tabular}{@{}l l l r r r r r r@{}}",
        r"\toprule",
        r"\textbf{Config} & \textbf{Optimizer / readout / $\beta$} & \textbf{bias init} & "
        r"\textbf{$\bar{R}$ ($\uparrow$)} & \textbf{SE} & \textbf{succ.} & \textbf{$n$} & "
        r"\textbf{$\Delta$ vs bench.} & \textbf{conf.} \\",
        r"\midrule",
    ]
    for i, r in enumerate(rows):
        if r[0].startswith("run-3.2.2 benchmark"):
            beta_cell = r"Adam / mse / $10^{2}$"
            delta_s, conf_s = "--- (ref.)", "---"
        else:
            beta_cell = f"${r[1].split(' / ')[-1]}$"
            beta_cell = r[1].rsplit(" / ", 1)
            beta_cell = f"{beta_cell[0]} / ${beta_cell[1]}$"
            d = r[3] - bench_m
            conf = phi(d / math.sqrt(r[4] ** 2 + bench_se ** 2))
            delta_s, conf_s = f"${d:+.2f}$", f"{round(100 * conf)}\\%"
        cell1 = beta_cell if not r[0].startswith("run-3.2.2 benchmark") else r"Adam / mse / $10^{2}$"
        lines.append(f"{r[0]} & {cell1} & {r[2]} & {rbar[i]} & {r[4]:.2f} & {succ_col[i]} & "
                     f"{r[6]} & {delta_s} & {conf_s} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(PLOTS, "validation_table.tex"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"validation_table.tex: {len(rows)} rows (bench n={bench_n})")


def main():
    """Load both sweeps' records and write the two tables."""
    sc3 = load_scores(os.path.join(RUN3, "data", "*", "local", "*.json"))
    sc4 = load_scores(os.path.join(RUN4, "data", "*", "local", "*.json"))
    print(f"loaded: 3.2.3 configs={len(sc3)} runs={sum(map(len, sc3.values()))}; "
          f"3.2.4 configs={len(sc4)} runs={sum(map(len, sc4.values()))}")
    make_reward_table(sc3)
    make_validation_table(sc4, sc3)


if __name__ == "__main__":
    main()
