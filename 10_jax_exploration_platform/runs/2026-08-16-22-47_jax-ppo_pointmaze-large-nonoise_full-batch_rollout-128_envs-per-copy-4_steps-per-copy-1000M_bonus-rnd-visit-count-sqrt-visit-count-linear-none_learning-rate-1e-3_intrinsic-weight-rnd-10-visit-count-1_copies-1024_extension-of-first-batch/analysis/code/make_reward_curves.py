"""Draw the training curve of each algorithm arm of the extension of train run 1.1.

This run holds one configuration per arm — the one that arm won with in the parent run — so the
"best configuration" the parent's script had to choose is here the arm's only configuration, and it
is still read through `code/aggregate.best_per_arm` rather than hardcoded, so the figure and the
development document's results table cannot name different configurations. A configuration's copies
may have been run as several chunks on several cards; `aggregate.curve_of` pools them per copy per
window, so the curve is the mean over all 1,024 copies and the band is plus or minus one standard
error over them.

The figure keeps the file name `pm_jax_run11_reward_curves.pdf`, because it fills the same slot in
the development document (\\Cref{fig:pm-jax-run11-curves}) — but it is written into THIS run folder,
so the parent's own plot survives untouched as the record of the shorter budget. The document's
figure now shows the thousand-million-step curves only: two budgets of one arm drawn in one panel
read as two algorithms.

Solid black is reserved for the best random-network-distillation configuration: it is the reference
the other arms are measured against. Every other arm has its own colour and its own line style, the
same pair the parent run gave it.

Run:
  PYTHONNOUSERSITE=1 /p/rlprojects/RND/.venvs/platform_jax/bin/python analysis/code/make_reward_curves.py
"""
import sys
from pathlib import Path

RUN_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(RUN_DIR / "code"))

import matplotlib                                                        # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                          # noqa: E402
import numpy as np                                                       # noqa: E402

from aggregate import best_per_arm, cell_table, curve_of                 # noqa: E402

PLOTS = RUN_DIR / "analysis" / "plots"

# (colour, line style, human name) per arm, identical to the parent run's assignment so the two
# runs' figures read the same way. Black is the distillation reference and is used by no other arm;
# every other arm also differs in line style, so the panel reads without colour.
ARM_STYLE = {
    "rnd_next_state": ("black", "-", "random network distillation"),
    "gt_position_velocity_sqrt": ("#0072B2", "--", r"oracle counts $1/\sqrt{n}$"),
    "gt_position_velocity_linear": ("#D55E00", "-.", "oracle counts $1/n$"),
    "none": ("#009E73", ":", "no bonus"),
}
ARM_ORDER = ["rnd_next_state", "gt_position_velocity_sqrt", "gt_position_velocity_linear", "none"]


def main() -> None:
    """Draw one panel: every arm's extended configuration, on the one environment of this run."""
    rows = cell_table(completed_only=False)
    best = best_per_arm(rows)
    if not best:
        raise SystemExit("no scored configuration yet: the run has written no phase-blocked window")

    PLOTS.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 1, figsize=(7.2, 4.4))
    for bonus in ARM_ORDER:
        if bonus not in best:
            continue
        row = best[bonus]
        colour, style, label = ARM_STYLE[bonus]
        curve = curve_of(bonus, row["learning_rate"], row["intrinsic_weight"], RUN_DIR)
        # before: env_steps_per_copy in units of steps (up to 1,000,038,400); after: millions,
        # which is the unit the run is planned and reported in
        steps = np.asarray(curve["env_steps_per_copy"], dtype=float) / 1e6
        mean = np.asarray(curve["mean_episode_return"], dtype=float)
        error = np.asarray(curve["standard_error"], dtype=float)
        weight_note = ("" if bonus == "none"
                       else rf", $\beta = 10^{{{int(round(np.log10(row['intrinsic_weight'])))}}}$")
        rate_note = rf"$\alpha = 10^{{{int(round(np.log10(row['learning_rate'])))}}}$"
        axes.plot(steps, mean, color=colour, linestyle=style, linewidth=1.2,
                  label=f"{label} ({rate_note}{weight_note}, n={row['copies']})")
        axes.fill_between(steps, mean - error, mean + error, color=colour, alpha=0.18,
                          linewidth=0)

    axes.set_xlabel("environment steps per copy (millions)")
    axes.set_ylabel("mean episode return")
    axes.set_title("PointMaze large, no reset noise — each arm's best configuration, 1000M steps")
    axes.grid(alpha=0.25, linewidth=0.6)
    axes.legend(loc="upper left", fontsize=8, framealpha=0.9)
    figure.tight_layout()
    for suffix in ("pdf", "png"):
        figure.savefig(PLOTS / f"pm_jax_run11_reward_curves.{suffix}", dpi=180)
    print(f"wrote {PLOTS / 'pm_jax_run11_reward_curves.pdf'} and its .png")


if __name__ == "__main__":
    main()
