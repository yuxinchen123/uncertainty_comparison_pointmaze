#!/usr/bin/env python
"""Regenerate the three interim-results tables of subsection 8.1 (point maze + ant maze train
run 1) inline in RND_development_document.tex, between the AUTO-GENERATED TABLE markers:

  - pm-am-run1-interim-status    : racing status, one row per (env, algorithm) cell + a total row
  - pm-am-run1-interim-pointmaze : per-env metrics tables of the 4 PointMaze envs (one block each)
  - pm-am-run1-interim-antmaze   : per-env metrics tables of the 4 AntMaze envs (one block each)

Only the \\begin{tabular}...\\end{tabular} blocks are replaced; captions, labels, sizing and the
surrounding dark-brown prose stay hand-edited in RND_development_document.tex (generate-latex-table skill convention).

All counting/scoring is IMPORTED from the run's monitoring modules (status_table,
env_metrics_tables, monitoring_report) and from slurm/build_queue + slurm/prune_controller — the
same source of truth as the 20-minute monitoring report — so the writeup tables can never drift
from the monitoring tables. Marking follows monitoring_report.MARK_SPEC (analysis-convention
rule): per metric column, per env block, best value bold ({\\boldmath...}), second best
\\underline{}; higher is better everywhere except steps-to-goal; the N column is never marked.

Usage:
  /p/rlprojects/RND/.venvs/exploration/bin/python make_all.py
(reads every completed per-run JSON of the sweep — takes a few minutes)
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MAIN_TEX = HERE.parent.parent / "RND_development_document.tex"
RUN_DIR = (HERE.parent.parent.parent / "train_runs" /
           "2026-07-23-01-35_run_8_1_pointmaze-antmaze-8env-start-bottom-left_sac__rnd-origsmall"
           "__gt-position-velocity__gt-position-maze-cell__gt-position-1m__beta-1e-3-to-1e4-x15"
           "_prune20-winner100-max300_1Mstep_default-max-episode_reward-shift-1_gamma0.99")
SWEEP_ID = "2026-07-23-02-05_pm-am-run1"

sys.path.insert(0, str(RUN_DIR / "20_mins_monitoring"))
sys.path.insert(0, str(RUN_DIR / "slurm"))
import build_queue          # noqa: E402  (ENV_SETUPS_RUN1 — env order, config source of truth)
import status_table         # noqa: E402  (build_rows — the racing-status counting)
import env_metrics_tables   # noqa: E402  (load_completed — the completed-record loader)
import monitoring_report    # noqa: E402  (metric_rows / MARK_SPEC — row building + marking spec)


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
            "(expect exactly 1 of each). Add the markers in RND_development_document.tex first."
        )
    before, _, rest = text.partition(begin)
    _, _, after = rest.partition(end)
    tex_path.write_text(f"{before}{begin}\n{tabular_block}\n{end}{after}")


def tt(name: str) -> str:
    """A snake_case identifier as \\texttt with escaped, breakable underscores (house style)."""
    return "\\texttt{" + name.replace("_", "\\_\\allowbreak ") + "}"


def latex_metric_rows(env_setup, by_key):
    """(rows, awaiting) for one env: monitoring_report.metric_rows re-rendered as LaTeX cells.
    Each row dict maps column key -> LaTeX cell string, plus _raw means for marking."""
    md_rows, awaiting = monitoring_report.metric_rows(env_setup, by_key)
    out = []
    for r in md_rows:
        raw = r["_raw"]
        # steps sub-count: recover it from the markdown cell's "(n=..)" suffix
        sub_n = None
        if "(n=" in r["steps"]:
            sub_n = int(r["steps"].split("(n=")[1].rstrip(")"))
        # the monitoring label is "<algorithm> — <knob>" (em dash); re-render for LaTeX
        algorithm, knob = r["label"].split(" — ", 1)
        latex_label = tt(algorithm) + " --- " + knob
        out.append({
            "label": latex_label,
            "reward": _remark(r["reward"], raw["reward"], 2),
            "reward100": _remark(r["reward100"], raw["reward100"], 2),
            "success": _remark(r["success"], raw["success"], 3),
            "steps": _remark(r["steps"], raw["steps"], 1, sub_n),
            "mcov": _remark(r["mcov"], raw["mcov"], 2),
            "cov1m": _remark(r["cov1m"], raw["cov1m"], 2),
            "N": r["N"],
            "_raw": raw,
        })
    return out, awaiting


def _remark(md_cell, raw_mean, prec, sub_n=None):
    """Convert one monitoring markdown cell ("m ± s", possibly with an "(n=..)" suffix) to the
    LaTeX cell. The markdown cell is the formatting source of truth (same rounding)."""
    if md_cell == "—":
        return "---"
    body = md_cell
    if sub_n is not None:
        body = body.split(" (n=")[0]
    mean_s, se_s = body.split(" ± ")
    s = f"${mean_s} \\pm {se_s}$"
    if sub_n is not None:
        s += f" (n={sub_n})"
    return s


def mark_latex_rows(rows):
    """In-place bold the best and underline the second-best LaTeX cell of each marked metric
    column WITHIN this env block (monitoring_report.mark_rows, LaTeX rendering). Ranking uses the
    display-rounded mean so ties match what the reader sees; a row with no value for a column is
    skipped for that column; N and the label are never marked."""
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


HEADS = {
    "label": "\\textbf{algorithm --- best configuration}",
    "reward": "\\textbf{\\shortstack[c]{whole-run\\\\reward $\\uparrow$}}",
    "reward100": "\\textbf{\\shortstack[c]{last-100\\\\reward}}",
    "success": "\\textbf{\\shortstack[c]{success\\\\rate}}",
    "steps": "\\textbf{\\shortstack[c]{steps to\\\\goal}}",
    "mcov": "\\textbf{\\shortstack[c]{maze-cell\\\\coverage \\%}}",
    "cov1m": "\\textbf{\\shortstack[c]{$1$\\,m\\\\coverage \\%}}",
    "N": "\\textbf{$N$}",
}


def build_family_tabular(env_setups, by_key, include_cov1m: bool) -> str:
    """One tabular holding the per-env metrics blocks of a maze family: the header row once, then
    one block per env (an env-name rule row + its ranked, marked algorithm rows). The PointMaze
    family drops the 1 m coverage column (identical to maze-cell coverage there --- the PointMaze
    maze cell IS 1 m); the AntMaze family keeps both (4 m cells vs 1 m squares)."""
    cols = ["label", "reward", "reward100", "success", "steps", "mcov"]
    if include_cov1m:
        cols.append("cov1m")
    cols.append("N")
    ncols = len(cols)
    colspec = "@{}>{\\raggedright\\arraybackslash}p{3.8cm} " + "r " * (ncols - 2) + "r@{}"
    lines = ["\\begin{tabular}{" + colspec + "}",
             "\\toprule",
             " &\n".join(HEADS[c] for c in cols) + " \\\\"]
    for env_setup in env_setups:
        rows, awaiting = latex_metric_rows(env_setup, by_key)
        mark_latex_rows(rows)
        lines.append("\\midrule")
        lines.append("\\multicolumn{" + str(ncols) + "}{@{}l}{\\textbf{" + tt(env_setup)
                     + "}} \\\\")
        for r in rows:
            lines.append(" & ".join(r[c] for c in cols) + " \\\\")
        if awaiting:
            lines.append("\\multicolumn{" + str(ncols) + "}{@{}l}{(awaiting a first completed "
                         "record: " + ", ".join(tt(a) for a in awaiting) + ")} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines)


def build_status_tabular() -> str:
    """The racing-status tabular: one row per (env, algorithm) cell (status_table.build_rows) plus
    the total row. Columns kept for the writeup: configurations, still racing, pruned, winner
    phase, completed runs (the queue's volatile pending/running/done counts are left to the
    monitoring snapshots)."""
    rows, total = status_table.build_rows(SWEEP_ID)
    lines = ["\\begin{tabular}{@{}l l r r r r r@{}}",
             "\\toprule",
             "\\textbf{environment} & \\textbf{algorithm} & \\textbf{configurations} & "
             "\\textbf{still racing} & \\textbf{pruned} & \\textbf{winner phase} & "
             "\\textbf{completed runs} \\\\",
             "\\midrule"]
    prev_env = None
    for r in rows:
        env_cell = tt(r["env"]) if r["env"] != prev_env else ""
        prev_env = r["env"]
        lines.append(" & ".join([env_cell, tt(r["algorithm"]), str(r["cfgs"]), str(r["racing"]),
                                 str(r["pruned"]), str(r["wcut"]), str(r["compl"])]) + " \\\\")
    lines.append("\\midrule")
    lines.append(" & ".join(["\\textbf{total}", "", "\\textbf{" + str(total["cfgs"]) + "}",
                             "\\textbf{" + str(total["racing"]) + "}",
                             "\\textbf{" + str(total["pruned"]) + "}",
                             "\\textbf{" + str(total["wcut"]) + "}",
                             "\\textbf{" + str(total["compl"]) + "}"]) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines), total


POINTMAZE_ENVS = [e for e in build_queue.ENV_SETUPS_RUN1 if e.startswith("PointMaze")]
ANTMAZE_ENVS = [e for e in build_queue.ENV_SETUPS_RUN1 if e.startswith("AntMaze")]


def main() -> None:
    status_block, total = build_status_tabular()
    inject_table(MAIN_TEX, "pm-am-run1-interim-status", status_block)
    print(f"[status] configurations={total['cfgs']} racing={total['racing']} "
          f"pruned={total['pruned']} winner_phase={total['wcut']} completed_runs={total['compl']} "
          f"failed={total['failed']}")

    print("[load] reading completed per-run records ...")
    by_key = env_metrics_tables.load_completed(SWEEP_ID)
    n_records = sum(len(v) for v in by_key.values())
    print(f"[load] {n_records} completed records across {len(by_key)} configurations")

    inject_table(MAIN_TEX, "pm-am-run1-interim-pointmaze",
                 build_family_tabular(POINTMAZE_ENVS, by_key, include_cov1m=False))
    inject_table(MAIN_TEX, "pm-am-run1-interim-antmaze",
                 build_family_tabular(ANTMAZE_ENVS, by_key, include_cov1m=True))
    print(f"Updated {MAIN_TEX}")


if __name__ == "__main__":
    main()
