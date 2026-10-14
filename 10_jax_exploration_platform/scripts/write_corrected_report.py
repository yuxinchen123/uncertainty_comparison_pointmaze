"""Generate the corrected-run table and two figures from its published aggregate only."""
import argparse
import json
from pathlib import Path

NAMES = {"rnd_next_state": "random network distillation",
         "gt_position_velocity_sqrt": r"oracle counts $1/\sqrt{n}$",
         "gt_position_velocity_linear": r"oracle counts $1/n$", "none": "no bonus"}
FIRST = [("rnd_next_state", 1e-3, 10), ("gt_position_velocity_sqrt", 1e-3, 1),
         ("gt_position_velocity_linear", 1e-3, 1), ("none", 1e-3, 0),
         ("rnd_next_state", 1e-4, 100), ("gt_position_velocity_sqrt", 1e-4, 10)]
SECOND = [("gt_position_velocity_sqrt", 1e-3, 1),
          ("gt_position_velocity_sqrt", 1e-4, 1), ("rnd_next_state", 1e-3, 100),
          ("rnd_next_state", 1e-4, 1000), ("rnd_next_state", 1e-3, 1),
          ("rnd_next_state", 1e-4, 1)]


def power(value):
    """Format the power-of-ten settings used by this selected experiment."""
    import math
    return rf"10^{{{int(round(math.log10(value)))}}}"


def draw(curves, settings, output, shape_plot=False):
    """Draw the same historical settings with corrected training and seed standard errors."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    colors = {"rnd_next_state": "black", "gt_position_velocity_sqrt": "#0072B2",
              "gt_position_velocity_linear": "#D55E00", "none": "#009E73"}
    figure, axes = plt.subplots(figsize=(7.6, 4.8))
    for bonus, rate, weight in settings:
        curve = curves[f"{bonus}:{rate:g}:{weight:g}"]
        steps = np.asarray([point["steps"] for point in curve]) / 1e6
        means = np.asarray([point["mean"] for point in curve])
        errors = np.asarray([point["standard_error"] for point in curve])
        color = "#E69F00" if shape_plot and bonus == "rnd_next_state" and weight == 1 else colors[bonus]
        style = ":" if rate == 1e-3 else "-"
        label = NAMES[bonus] + rf", $\alpha={power(rate)}$"
        if bonus != "none":
            label += rf", $\beta={power(weight)}$"
        axes.plot(steps, means, color=color, linestyle=style, linewidth=1.2, label=label)
        axes.fill_between(steps, means - errors, means + errors, color=color, alpha=.13,
                          linewidth=0)
    axes.set(xscale="log", xlabel="environment steps per copy (millions, logarithmic)",
             ylabel="mean episode return", title="Corrected intrinsic returns: 100M steps per copy")
    axes.grid(alpha=.25, which="both")
    axes.legend(loc="upper left", fontsize=7, framealpha=.9)
    figure.tight_layout()
    for extension in ("pdf", "png"):
        figure.savefig(output.with_suffix("." + extension), dpi=180)
    plt.close(figure)


def table(rows):
    """Render all completed settings, marking the best and second values within this block."""
    fields = ("whole_run_reward", "last_window_reward", "success_rate", "coverage_percent")
    rankings = {field: sorted({row[field] for row in rows}, reverse=True) for field in fields}
    lines = [r"\begin{tabular}{@{}p{4.4cm}rrrrr@{}}", r"\toprule",
             r"\textbf{algorithm and settings} & \shortstack{whole-run\\reward $\uparrow$} &"
             r"\shortstack{final-window\\reward} & \shortstack{ever\\reached goal} &"
             r"\shortstack{coverage\\(\%)} & $N$ \\", r"\midrule"]
    for row in sorted(rows, key=lambda item: item["whole_run_reward"], reverse=True):
        name = NAMES[row["bonus"]]
        name += rf"\newline $\alpha={power(row['learning_rate'])}$"
        if row["bonus"] != "none":
            name += rf", $\beta={power(row['intrinsic_weight'])}$"
        values = []
        for field in fields:
            value = row[field]
            number = f"{value:.3f}"
            if field != "success_rate":
                number += rf" \pm {row[field + '_standard_error']:.3f}"
            cell = "$" + number + "$"
            if value == rankings[field][0]:
                cell = "{\\boldmath " + cell + "}"
            elif len(rankings[field]) > 1 and value == rankings[field][1]:
                cell = "\\underline{" + cell + "}"
            values.append(cell)
        lines.append(name + " & " + " & ".join(values) + f" & {row['copies']} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def inject(text, name, content):
    """Replace exactly one named generated block without modifying captions or other prose."""
    begin = f"% >>> AUTO-GENERATED TABLE START: {name}"
    end = f"% <<< AUTO-GENERATED TABLE END: {name}"
    if text.count(begin) != 1 or text.count(end) != 1:
        raise ValueError(f"missing or duplicate markers for {name}")
    left, tail = text.split(begin)
    _, right = tail.split(end)
    return left + begin + "\n" + content + "\n" + end + right


def main():
    """Require all requested settings, generate artifacts, and insert the completed results."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--tex", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads((args.run_dir / "summary.json").read_text())
    if summary["units_complete"] != 11 or len(summary["cells"]) != 11:
        raise ValueError("report requires all 11 selected configurations to complete")
    plots = args.run_dir / "analysis/plots"
    draw(summary["curves"], FIRST, plots / "corrected_reward_curves")
    draw(summary["curves"], SECOND, plots / "corrected_rate_weight_curves", True)
    # Select one setting per algorithm by the prespecified whole-run mean, within this fixed subset.
    best_per_algorithm = {}
    for row in summary["cells"]:
        previous = best_per_algorithm.get(row["bonus"])
        if previous is None or row["whole_run_reward"] > previous["whole_run_reward"]:
            best_per_algorithm[row["bonus"]] = row
    text = inject(args.tex.read_text(), "pm-jax-corrected-results",
                  table(list(best_per_algorithm.values())))
    args.tex.write_text(text)
    best = max(summary["cells"], key=lambda row: row["whole_run_reward"])
    (args.run_dir / "analysis/analysis.md").write_text(
        "Generated by scripts/write_corrected_report.py from summary.json.\n\n"
        "All 11 selected configurations completed with 128 paired seeds each. "
        f"The highest whole-run mean reward was {best['whole_run_reward']:.3f} "
        f"for {NAMES[best['bonus']]} at learning rate {best['learning_rate']:g} "
        f"and intrinsic weight {best['intrinsic_weight']:g}. "
        "The 100-million-step budget does not establish the later peaks observed in the "
        "historical 1000-million-step experiments.\n")
    print(f"updated {args.tex}")


if __name__ == "__main__":
    main()
