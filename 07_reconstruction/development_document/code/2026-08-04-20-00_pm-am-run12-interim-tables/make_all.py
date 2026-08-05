#!/usr/bin/env python
"""Regenerate the two interim-results tables of subsubsection 8.1.2 (train run 1.2) inline in
RND_development_document.tex, between the AUTO-GENERATED TABLE markers:

  - pm-am-trainrun12-status  : stage-1 status, one row per (environment, arm) + a total row
  - pm-am-trainrun12-interim : per-environment best-configuration tables (one block per env)

Only the \\begin{tabular}...\\end{tabular} blocks are replaced; captions, labels, sizing and the
surrounding prose stay hand-edited in RND_development_document.tex (generate-latex-table skill).

All counting/scoring is IMPORTED from the run's monitoring modules (status_table,
env_metrics_tables, monitoring_report) — the same source of truth as the 20-minute monitoring
report — so the writeup tables can never drift from the monitoring tables. Marking follows
monitoring_report.MARK_SPEC (analysis-convention rule): within each environment block the best
whole-run reward is bold ({\\boldmath...}) and the second best \\underline{}d; the completed-seed
and verdict columns are never marked.

Usage:
  /p/rlprojects/RND/.venvs/exploration/bin/python make_all.py
(reads every completed per-run JSON of the sweep — a few minutes at full sweep size)
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MAIN_TEX = HERE.parent.parent / "RND_development_document.tex"
RUN_DIR = (HERE.parent.parent.parent / "train_runs" /
           "2026-08-01-01-44_run_8_1_2_antmaze-umaze-medium-bottom-left_sac__rnd-origsmall-beta1e4"
           "-3e3-10M-100seed__alg1-sgd-l2-biasnormal0.5__alg2.1-ratio__alg2.2-normloss__alg2.3-"
           "layernorm__lr-1e-3-1e-2_beta-1e-3-to-1e4-x15_1M-prune30-race100-frozenbar_gamma0.99")
SWEEP_ID = "2026-08-01-02-03_run812"

sys.path.insert(0, str(RUN_DIR / "20_mins_monitoring"))
sys.path.insert(0, str(RUN_DIR / "slurm"))
import build_queue          # noqa: E402  (ENV_SETUPS_RUN12 — env order, config source of truth)
import status_table         # noqa: E402  (build_rows / decision_verdicts — status counting)
import env_metrics_tables   # noqa: E402  (load_completed / env_rows / frozen_bars / above_bar_counts)
import monitoring_report    # noqa: E402  (MARK_SPEC — the marking spec)

# writeup display name per queue arm tag (the writeup names the arms "algorithm 1", "algorithm 2.1"
# ...; the queue uses the short tags)
ARM_DISPLAY = {"baseline": "RND baseline", "alg1": "algorithm 1", "alg2.1": "algorithm 2.1",
               "alg2.2": "algorithm 2.2", "alg2.3": "algorithm 2.3"}


def inject_table(tex_path: Path, table_name: str, tabular_block: str) -> None:
    """Replace the body between the AUTO-GENERATED TABLE markers for `table_name`.

    Hard-fails if either marker is missing or duplicated, so a typo in the marker
    name surfaces immediately rather than silently no-op'ing.
    """
    begin = f"% >>> AUTO-GENERATED TABLE START: {table_name}"
    end = f"% <<< AUTO-GENERATED TABLE END: {table_name}"
    text = tex_path.read_text()
    if text.count(begin) != 1 or text.count(end) != 1:
        raise SystemExit(
            f"Marker mismatch for table {table_name!r} in {tex_path}: "
            f"BEGIN count={text.count(begin)}, END count={text.count(end)} "
            "(expect exactly 1 of each). Add the markers in RND_development_document.tex first.")
    before, _, rest = text.partition(begin)
    _, _, after = rest.partition(end)
    tex_path.write_text(f"{before}{begin}\n{tabular_block}\n{end}{after}")


def tt(name: str) -> str:
    """A snake_case identifier as \\texttt with escaped, breakable underscores (house style)."""
    return "\\texttt{" + name.replace("_", "\\_\\allowbreak ") + "}"


def latex_label(md_label: str) -> str:
    """One monitoring row label re-rendered for the writeup.
    before: "alg2.2 — lr 0.01, bonus-weight 3"        after: "algorithm 2.2 --- lr 0.01, bonus-weight 3"
    before: "baseline — lr 0.0001, bonus-weight 10000" after: "RND baseline --- bonus-weight 10000 ($10^{7}$ steps)"
    (the baseline's lr is its fixed adam rate, not a swept knob, so it is dropped; its step count is
    stated because its rows are 10M runs while every other row is a 1M run)."""
    arm, knobs = md_label.split(" — ", 1)
    if arm == "baseline":
        beta = knobs.split("bonus-weight ")[1]
        return f"{ARM_DISPLAY[arm]} --- bonus-weight {beta} ($10^{{7}}$ steps)"
    return f"{ARM_DISPLAY[arm]} --- {knobs}"


def latex_cell(md_cell: str) -> str:
    """One monitoring "mean ± se" markdown cell as a LaTeX math cell ("—" stays a --- dash).
    before: "-672.84 ± 1.52"   after: "$-672.84 \\pm 1.52$" """
    if md_cell == "—":
        return "---"
    mean_s, se_s = md_cell.split(" ± ")
    return f"${mean_s} \\pm {se_s}$"


def latex_metric_rows(env_setup, by_key, verdicts):
    """(rows, awaiting) for one env: env_metrics_tables.env_rows re-rendered as LaTeX cells."""
    md_rows, awaiting = env_metrics_tables.env_rows(env_setup, by_key, verdicts)
    out = []
    for r in md_rows:
        out.append({
            "label": latex_label(r["label"]),
            "reward": latex_cell(r["reward"]),
            "reward100": latex_cell(r["reward100"]),
            "N": r["N"],
            "verdict": r["verdict"],
            "_raw": r["_raw"],
        })
    return out, [ARM_DISPLAY[a] for a in awaiting]


def mark_latex_rows(rows):
    """In-place bold the best and underline the second-best reward cell WITHIN this env block
    (monitoring_report.MARK_SPEC, LaTeX rendering; display-rounded means so ties match the page)."""
    for key, higher, prec in monitoring_report.MARK_SPEC:
        present = [(i, round(r["_raw"][key], prec)) for i, r in enumerate(rows)
                   if r["_raw"][key] is not None]
        if not present:
            continue
        vals = sorted({v for _, v in present}, reverse=higher)
        best = vals[0]
        second = vals[1] if len(vals) > 1 else None
        for i, v in present:
            if v == best:
                rows[i][key] = "{\\boldmath" + rows[i][key] + "}"
            elif second is not None and v == second:
                rows[i][key] = "\\underline{" + rows[i][key] + "}"


def bar_row(env_setup, by_key, ncols) -> str:
    """The full-width frozen-bar line under one env block, rebuilt from the modules' own numbers.
    before: bar=-689.7404, beta="10000", k=2, total=120
    after:  \\multicolumn{4}{@{}l}{\\emph{frozen bar $-689.7404$ (bonus-weight 10000); ...}} \\\\"""
    bar, beta = env_metrics_tables.frozen_bars()[env_setup]
    k, total = env_metrics_tables.above_bar_counts(env_setup, by_key, bar)
    return ("\\multicolumn{" + str(ncols) + "}{@{}l}{\\emph{frozen bar $" + f"{bar:.4f}"
            + "$ (bonus-weight " + beta + "); " + str(k) + " of " + str(total)
            + " configurations at or above it so far}} \\\\")


def build_metrics_tabular(by_key, verdicts) -> str:
    """One tabular with both environments' blocks: header once, then per env a rule row, the ranked
    and marked best-configuration rows, the frozen-bar line, and the awaiting note if any."""
    cols = ["label", "reward", "reward100", "N", "verdict"]
    heads = {"label": "\\textbf{arm --- best configuration so far}",
             "reward": "\\textbf{\\shortstack[c]{whole-run\\\\reward $\\bar{R}$ $\\uparrow$}}",
             "reward100": "\\textbf{\\shortstack[c]{final\\\\reward $\\bar{R}_{100}$}}",
             "N": "\\textbf{\\shortstack[c]{completed\\\\seeds $n$}}",
             "verdict": "\\textbf{\\shortstack[c]{verdict\\\\so far}}"}
    lines = ["\\begin{tabular}{@{}>{\\raggedright\\arraybackslash}p{5.6cm} r r r l@{}}",
             "\\toprule",
             " &\n".join(heads[c] for c in cols) + " \\\\"]
    for env_setup in build_queue.ENV_SETUPS_RUN12:
        rows, awaiting = latex_metric_rows(env_setup, by_key, verdicts)
        mark_latex_rows(rows)
        lines.append("\\midrule")
        lines.append("\\multicolumn{5}{@{}l}{\\textbf{" + tt(env_setup) + "}} \\\\")
        for r in rows:
            lines.append(" & ".join(r[c] for c in cols) + " \\\\")
        lines.append(bar_row(env_setup, by_key, 5))
        if awaiting:
            lines.append("\\multicolumn{5}{@{}l}{(awaiting a first completed record: "
                         + ", ".join(awaiting) + ")} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines)


def build_status_tabular() -> str:
    """The stage-1 status tabular: one row per (environment, arm) (status_table.build_rows) plus the
    total row. Columns kept for the writeup: configurations, undecided, pruned, survivors, completed
    runs (the volatile pending/running/failed counts stay in the monitoring snapshots)."""
    rows, total = status_table.build_rows(SWEEP_ID)
    lines = ["\\begin{tabular}{@{}l l r r r r r@{}}",
             "\\toprule",
             "\\textbf{environment} & \\textbf{arm} & \\textbf{configurations} & "
             "\\textbf{undecided} & \\textbf{pruned} & \\textbf{survivors} & "
             "\\textbf{completed runs} \\\\",
             "\\midrule"]
    prev_env = None
    for r in rows:
        env_cell = tt(r["env"]) if r["env"] != prev_env else ""
        prev_env = r["env"]
        lines.append(" & ".join([env_cell, ARM_DISPLAY[r["arm"]], str(r["cfgs"]),
                                 str(r["undecided"]), str(r["pruned"]), str(r["survivors"]),
                                 str(r["compl"])]) + " \\\\")
    lines.append("\\midrule")
    lines.append(" & ".join(["\\textbf{total}", "", "\\textbf{" + str(total["cfgs"]) + "}",
                             "\\textbf{" + str(total["undecided"]) + "}",
                             "\\textbf{" + str(total["pruned"]) + "}",
                             "\\textbf{" + str(total["survivors"]) + "}",
                             "\\textbf{" + str(total["compl"]) + "}"]) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines), total


def main() -> None:
    status_block, total = build_status_tabular()
    inject_table(MAIN_TEX, "pm-am-trainrun12-status", status_block)
    print(f"[status] configurations={total['cfgs']} undecided={total['undecided']} "
          f"pruned={total['pruned']} survivors={total['survivors']} completed={total['compl']}")

    print("[load] reading completed per-run records ...")
    by_key = env_metrics_tables.load_completed(SWEEP_ID)
    verdicts = status_table.decision_verdicts(SWEEP_ID)
    n_records = sum(len(v) for v in by_key.values())
    print(f"[load] {n_records} completed records across {len(by_key)} configurations")

    inject_table(MAIN_TEX, "pm-am-trainrun12-interim", build_metrics_tabular(by_key, verdicts))
    print(f"Updated {MAIN_TEX}")


if __name__ == "__main__":
    main()
