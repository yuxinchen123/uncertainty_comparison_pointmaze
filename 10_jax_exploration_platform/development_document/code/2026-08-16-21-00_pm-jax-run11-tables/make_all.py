#!/usr/bin/env python
"""Regenerate the best-configuration results table of subsection 1.1 inline in
`platform_development_document.tex`, between the AUTO-GENERATED TABLE markers:

  - pm-jax-run11-results : one row per algorithm arm at its best configuration, on the one
                           environment of this run

Only the `\\begin{tabular}...\\end{tabular}` block is replaced; the caption, the label, the
`\\resizebox` and the surrounding prose stay hand-edited in the `.tex`.

Every number is IMPORTED from the run's own aggregation module (`<run>/code/aggregate.py`) — the
same module the 20-minute status table and the curve figure read — so the writeup cannot drift from
what the monitor reports. Nothing here loads a shard, scores a copy, or ranks a configuration.

Marking follows the analysis-convention rule: per metric column, best value bold, second best
underlined; higher is better for every column that has numbers here; the N column is never marked.

Usage:
  PYTHONNOUSERSITE=1 /p/rlprojects/RND/.venvs/platform_jax/bin/python make_all.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MAIN_TEX = HERE.parent.parent / "platform_development_document.tex"
PLATFORM_ROOT = HERE.parent.parent.parent
RUN_DIR = (PLATFORM_ROOT / "runs" /
           "2026-08-16-20-39_jax-ppo_pointmaze-large-nonoise_full-batch_rollout-128_envs-per-copy-4"
           "_steps-per-copy-10M_bonus-rnd-visit-count-sqrt-visit-count-linear-none"
           "_learning-rate-1e-3-1e-4-1e-5_intrinsic-weight-1e-5-to-1e5_copies-per-cell-256"
           "_first-batch")

sys.path.insert(0, str(RUN_DIR / "code"))
import aggregate  # noqa: E402  (cell_table / best_per_arm — this run's scoring, imported not copied)

ENV_SPEC = "pointmaze_large_cont400_nonoise@1"

# the order the arms are named in everywhere else in this run, and their document labels
ARM_LABEL = {
    "rnd_next_state": "rnd\\_\\allowbreak next\\_\\allowbreak state",
    "gt_position_velocity_sqrt": "gt\\_\\allowbreak position\\_\\allowbreak velocity\\_\\allowbreak sqrt",
    "gt_position_velocity_linear": "gt\\_\\allowbreak position\\_\\allowbreak velocity\\_\\allowbreak linear",
    "none": "no\\_\\allowbreak exploration",
}

# (column key, higher is better) for the columns that carry numbers and are marked
MARK_SPEC = [("whole_run_reward", True), ("last_window_reward", True),
             ("success_rate", True), ("coverage_percent", True)]


def inject_table(tex_path: Path, table_name: str, tabular_block: str) -> None:
    """Replace the body between the AUTO-GENERATED TABLE markers for `table_name`.

    Hard-fails when either marker is missing or appears twice, so a typo in the name surfaces
    immediately instead of silently doing nothing.
    """
    begin = f"% >>> AUTO-GENERATED TABLE START: {table_name}"
    end = f"% <<< AUTO-GENERATED TABLE END: {table_name}"
    text = tex_path.read_text()
    if text.count(begin) != 1 or text.count(end) != 1:
        raise SystemExit(
            f"Marker mismatch for table {table_name!r} in {tex_path}: "
            f"BEGIN count={text.count(begin)}, END count={text.count(end)} (expect exactly 1 of "
            "each). Add the markers in platform_development_document.tex first.")
    before, _, rest = text.partition(begin)
    _, _, after = rest.partition(end)
    tex_path.write_text(f"{before}{begin}\n{tabular_block}\n{end}{after}")


def power_of_ten(value: float) -> str:
    """A swept value's exponent, for use INSIDE a math span; the caller adds the dollar signs.

    before: 1e-05 ; after: "10^{-5}"
    """
    from math import log10
    return f"10^{{{int(round(log10(value)))}}}"


def row_label(row: dict) -> str:
    """The arm's name plus the configuration that won, spelled out rather than abbreviated."""
    weight = ("no intrinsic weight" if row["bonus"] == "none"
              else f"$\\beta = {power_of_ten(row['intrinsic_weight'])}$")
    return (f"\\texttt{{{ARM_LABEL[row['bonus']]}}}\\newline "
            f"\\small learning rate ${power_of_ten(row['learning_rate'])}$, {weight}")


def cell(mean, error, precision: int) -> str:
    """One mean-and-standard-error cell; a metric with no value prints an em dash, never blank."""
    if mean is None:
        return "---"
    return f"${mean:.{precision}f} \\pm {error:.{precision}f}$"


def mark(rows: list, cells: list) -> None:
    """Bold the best and underline the second best of each marked column, in place.

    before: three rows whose whole-run reward cells read $2.10 \\pm 0.05$, $1.80 ...$, $0.40 ...$
    after:  the first is wrapped in \\textbf{}, the second in \\underline{}, the third untouched.
    Ranking uses the raw mean; ties share the mark; a row with no value is skipped for that column.
    """
    for key, higher in MARK_SPEC:
        present = [(index, row[key]) for index, row in enumerate(rows) if row[key] is not None]
        if not present:
            continue
        values = sorted({value for _, value in present}, reverse=higher)
        best = values[0]
        second = values[1] if len(values) > 1 else None
        for index, value in present:
            # \textbf around a math span leaves the math unbolded, so the best cell uses \boldmath,
            # which switches the math font itself; the group keeps the switch inside the cell
            if value == best:
                cells[index][key] = f"{{\\boldmath {cells[index][key]}}}"
            elif second is not None and value == second:
                cells[index][key] = f"\\underline{{{cells[index][key]}}}"


def build_results_table() -> str:
    """The tabular block: one environment banner and one row per arm, best configuration first."""
    rows = list(aggregate.best_per_arm(aggregate.cell_table(RUN_DIR)).values())
    rows.sort(key=lambda row: row["whole_run_reward"], reverse=True)

    lines = [
        "\\begin{tabular}{@{}>{\\raggedright\\arraybackslash}p{4.6cm} r r r r r r@{}}",
        "\\toprule",
        "\\textbf{algorithm --- best configuration} &",
        "\\textbf{\\shortstack[c]{whole-run\\\\reward $\\uparrow$}} &",
        "\\textbf{\\shortstack[c]{last-window\\\\reward}} &",
        "\\textbf{\\shortstack[c]{success\\\\rate}} &",
        "\\textbf{\\shortstack[c]{maze-cell\\\\coverage \\%}} &",
        "\\textbf{\\shortstack[c]{steps\\\\to goal}} &",
        "\\textbf{$N$} \\\\",
        "\\midrule",
        f"\\multicolumn{{7}}{{@{{}}l}}{{\\textbf{{\\texttt{{"
        f"{ENV_SPEC.replace('_', chr(92) + '_' + chr(92) + 'allowbreak ')}}}}}}} \\\\",
    ]
    if not rows:
        lines += ["\\multicolumn{7}{@{}l}{no completed unit yet --- no configuration is scored} \\\\",
                  "\\bottomrule", "\\end{tabular}"]
        return "\n".join(lines)

    cells = [{
        "whole_run_reward": cell(row["whole_run_reward"], row["whole_run_reward_standard_error"], 3),
        "last_window_reward": cell(row["last_window_reward"],
                                   row["last_window_reward_standard_error"], 3),
        "success_rate": f"${row['success_rate']:.3f}$",
        "coverage_percent": cell(row["coverage_percent"], row["coverage_percent_standard_error"], 2),
    } for row in rows]
    mark(rows, cells)
    for row, cell_row in zip(rows, cells):
        lines.append(
            f"{row_label(row)} & {cell_row['whole_run_reward']} & "
            f"{cell_row['last_window_reward']} & {cell_row['success_rate']} & "
            f"{cell_row['coverage_percent']} & not recorded & ${row['copies']}$ \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def main() -> None:
    """Write the results table into the document and say what went in."""
    block = build_results_table()
    inject_table(MAIN_TEX, "pm-jax-run11-results", block)
    print(f"spliced pm-jax-run11-results into {MAIN_TEX} ({block.count(chr(92) + chr(92)) } rows "
          "of tabular material)")
    print(block)


if __name__ == "__main__":
    main()
