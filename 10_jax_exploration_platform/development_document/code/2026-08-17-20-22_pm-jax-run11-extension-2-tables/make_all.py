#!/usr/bin/env python
"""Regenerate the best-configuration results table of subsection 1.1 inline in
`platform_development_document.tex`, between the AUTO-GENERATED TABLE markers:

  - pm-jax-run11-results : one row per algorithm arm at its best configuration in the ORIGINAL
                           sweep; a double rule; the four arms as the FIRST extension re-ran the
                           configurations that sweep chose; a second double rule; the two arms as
                           the SECOND extension's own 1000M-step sweep of their whole grid chooses
                           them

This is the second extension's generator. The two before it,
`2026-08-16-21-00_pm-jax-run11-tables/` and `2026-08-16-22-27_pm-jax-run11-extension-tables/`, are
left in place as the record of what the table said before each extension ran; neither is called any
more.

Every number is IMPORTED from the run it belongs to, through that run's own `code/aggregate.py`, so
the writeup cannot drift from what each run's own monitor reports. Nothing here loads a shard,
scores a copy, or ranks a configuration. The three modules have the same file name, so they are
loaded by path under three different module names.

Marking follows the analysis-convention rule: per metric column, best value bold, second best
underlined, higher is better for every column that carries numbers, the N column never marked. It
is computed WITHIN each block: the blocks are different step budgets over different copy counts, so
marking one against another would compare measurements that are not comparable.

Usage:
  PYTHONNOUSERSITE=1 /p/rlprojects/RND/.venvs/platform_jax/bin/python make_all.py
"""
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MAIN_TEX = HERE.parent.parent / "platform_development_document.tex"
PLATFORM_ROOT = HERE.parent.parent.parent
RUNS = PLATFORM_ROOT / "runs"

ORIGINAL_RUN = RUNS / (
    "2026-08-16-20-39_jax-ppo_pointmaze-large-nonoise_full-batch_rollout-128_envs-per-copy-4"
    "_steps-per-copy-10M_bonus-rnd-visit-count-sqrt-visit-count-linear-none"
    "_learning-rate-1e-3-1e-4-1e-5_intrinsic-weight-1e-5-to-1e5_copies-per-cell-256_first-batch")
FIRST_EXTENSION_RUN = RUNS / (
    "2026-08-16-22-47_jax-ppo_pointmaze-large-nonoise_full-batch_rollout-128_envs-per-copy-4"
    "_steps-per-copy-1000M_bonus-rnd-visit-count-sqrt-visit-count-linear-none_learning-rate-1e-3"
    "_intrinsic-weight-rnd-10-visit-count-1_copies-1024_extension-of-first-batch")
SECOND_EXTENSION_RUN = RUNS / (
    "2026-08-17-10-45_jax-ppo_pointmaze-large-nonoise_full-batch_rollout-128_envs-per-copy-4"
    "_steps-per-copy-1000M_bonus-rnd-next-state-visit-count-sqrt"
    "_learning-rate-1e-3-1e-4-1e-5_intrinsic-weight-1e-5-to-1e5_copies-per-cell-128"
    "_extension-2-of-first-batch")

ENV_SPEC = "pointmaze_large_cont400_nonoise@1"

# the order the arms are named in everywhere else in this run family, and their document labels
ARM_LABEL = {
    "rnd_next_state": "rnd\\_\\allowbreak next\\_\\allowbreak state",
    "gt_position_velocity_sqrt": "gt\\_\\allowbreak position\\_\\allowbreak velocity\\_\\allowbreak sqrt",
    "gt_position_velocity_linear": "gt\\_\\allowbreak position\\_\\allowbreak velocity\\_\\allowbreak linear",
    "none": "no\\_\\allowbreak exploration",
}

# (column key, higher is better) for the columns that carry numbers and are marked
MARK_SPEC = [("whole_run_reward", True), ("last_window_reward", True),
             ("success_rate", True), ("coverage_percent", True)]


def load_run_module(run_dir: Path, module_name: str):
    """Import one run's own `code/aggregate.py` under a name of our choosing.

    The runs' modules are all called `aggregate`, so a plain `import aggregate` would return
    whichever was imported first and silently score one run with another's module.

    before: run_dir = <the second extension>, module_name = "aggregate_extension_2"
    after:  sys.modules["aggregate_extension_2"] is that run's module, with its own RUN_DIR
    """
    path = run_dir / "code" / "aggregate.py"
    if not path.exists():
        raise SystemExit(f"no aggregation module at {path}")
    # the module inserts the platform's scripts folder on sys.path at import time, and imports
    # `aggregate_run` lazily inside aggregate(), which this generator never calls
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


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


def copy_count(copies: int) -> str:
    """The N column's value, with the document's thousands separator.

    before: 1024 ; after: "$1{,}024$" — a bare comma in math mode sets the wrong spacing, so the
    document writes every thousands separator as a braced group.
    """
    return "$" + f"{copies:,}".replace(",", "{,}") + "$"


def cell(mean, error, precision: int) -> str:
    """One mean-and-standard-error cell; a metric with no value prints an em dash, never blank."""
    if mean is None:
        return "---"
    return f"${mean:.{precision}f} \\pm {error:.{precision}f}$"


def mark(rows: list, cells: list) -> None:
    """Bold the best and underline the second best of each marked column, in place.

    before: three rows whose whole-run reward cells read $2.10 \\pm 0.05$, $1.80 ...$, $0.40 ...$
    after:  the first is wrapped in \\boldmath, the second in \\underline{}, the third untouched.
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


def block_lines(module, run_dir: Path) -> list:
    """One run's rows, one per arm at its best configuration, sorted and marked within themselves."""
    rows = list(module.best_per_arm(module.cell_table(run_dir)).values())
    if not rows:
        return ["\\multicolumn{7}{@{}l}{no completed unit yet --- no configuration is scored} \\\\"]
    rows.sort(key=lambda row: row["whole_run_reward"], reverse=True)
    cells = [{
        "whole_run_reward": cell(row["whole_run_reward"], row["whole_run_reward_standard_error"], 3),
        "last_window_reward": cell(row["last_window_reward"],
                                   row["last_window_reward_standard_error"], 3),
        "success_rate": f"${row['success_rate']:.3f}$",
        "coverage_percent": cell(row["coverage_percent"], row["coverage_percent_standard_error"], 2),
    } for row in rows]
    mark(rows, cells)
    return [
        f"{row_label(row)} & {cell_row['whole_run_reward']} & "
        f"{cell_row['last_window_reward']} & {cell_row['success_rate']} & "
        f"{cell_row['coverage_percent']} & not recorded & {copy_count(row['copies'])} \\\\"
        for row, cell_row in zip(rows, cells)]


def build_results_table() -> str:
    """The tabular block: the sweep, a double rule, the first extension, a double rule, the second."""
    original = load_run_module(ORIGINAL_RUN, "aggregate_original")
    first = load_run_module(FIRST_EXTENSION_RUN, "aggregate_extension_1")
    second = load_run_module(SECOND_EXTENSION_RUN, "aggregate_extension_2")
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
    lines += block_lines(original, ORIGINAL_RUN)
    # first deliberate double rule: below it, the four configurations the 10M-step sweep chose,
    # re-run at 1000M steps with 1,024 copies each
    lines += ["\\midrule\\midrule"]
    lines += block_lines(first, FIRST_EXTENSION_RUN)
    # second deliberate double rule: below it, the two arms whose WHOLE grid was re-run at 1000M
    # steps, each shown at the configuration that grid chooses; the caption says so
    lines += ["\\midrule\\midrule"]
    lines += block_lines(second, SECOND_EXTENSION_RUN)
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def main() -> None:
    """Write the results table into the document and say what went in."""
    block = build_results_table()
    inject_table(MAIN_TEX, "pm-jax-run11-results", block)
    print(f"spliced pm-jax-run11-results into {MAIN_TEX}")
    print(block)


if __name__ == "__main__":
    main()
