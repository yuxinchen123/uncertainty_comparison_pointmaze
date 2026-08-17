"""Draw the training curve of the best configuration of each algorithm arm of train run 1.1.

The best configuration is chosen by the same score that ranks the development document's results
table — `code/aggregate.best_per_arm` — never re-derived here. The curve is the mean episode return
over that configuration's 256 copies at each recorded window, and the band is plus or minus one
standard error over those copies.

Solid black is reserved for the best random-network-distillation configuration: it is the reference
the other arms are measured against. Every other arm has its own colour and its own line style,
fixed across the run's figures.

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

from aggregate import best_per_arm, cell_copies, cell_curve, cell_table, read_units  # noqa: E402

PLOTS = RUN_DIR / "analysis" / "plots"

# (colour, line style, human name) per arm. Black is the distillation reference and is used by no
# other arm; every other arm also differs in line style, so the panel reads without colour.
ARM_STYLE = {
    "rnd_next_state": ("black", "-", "random network distillation"),
    "gt_position_velocity_sqrt": ("#0072B2", "--", r"oracle counts $1/\sqrt{n}$"),
    "gt_position_velocity_linear": ("#D55E00", "-.", "oracle counts $1/n$"),
    "none": ("#009E73", ":", "no bonus"),
}
ARM_ORDER = ["rnd_next_state", "gt_position_velocity_sqrt", "gt_position_velocity_linear", "none"]


def curve_of(unit_id: str, rate: float, weight: float, units: dict) -> dict:
    """The learning curve of one (arm, learning rate, intrinsic weight) cell of one unit."""
    unit = units[unit_id]
    members = cell_copies(unit)[(rate, weight)]
    return cell_curve(unit, members)


def main() -> None:
    """Draw one panel: the best configuration of every arm, on the one environment of this run."""
    units = read_units()
    rows = cell_table(completed_only=False)
    best = best_per_arm(rows)
    if not best:
        raise SystemExit("no scored cell yet: the run has written no phase-blocked window")

    PLOTS.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 1, figsize=(7.2, 4.4))
    for bonus in ARM_ORDER:
        if bonus not in best:
            continue
        row = best[bonus]
        colour, style, label = ARM_STYLE[bonus]
        curve = curve_of(row["unit_id"], row["learning_rate"], row["intrinsic_weight"], units)
        # before: env_steps_per_copy in units of steps (up to 10,035,200); after: millions, which
        # is the unit the run is planned and reported in
        steps = np.asarray(curve["env_steps_per_copy"], dtype=float) / 1e6
        mean = np.asarray(curve["mean_episode_return"], dtype=float)
        error = np.asarray(curve["standard_error"], dtype=float)
        weight_note = ("" if bonus == "none"
                       else rf", $\beta = 10^{{{int(round(np.log10(row['intrinsic_weight'])))}}}$")
        rate_note = rf"$\alpha = 10^{{{int(round(np.log10(row['learning_rate'])))}}}$"
        axes.plot(steps, mean, color=colour, linestyle=style, linewidth=1.8,
                  label=f"{label} ({rate_note}{weight_note}, n={row['copies']})")
        axes.fill_between(steps, mean - error, mean + error, color=colour, alpha=0.18,
                          linewidth=0)

    axes.set_xlabel("environment steps per copy (millions)")
    axes.set_ylabel("mean episode return")
    axes.set_title("PointMaze large, no reset noise — best configuration of each arm")
    axes.grid(alpha=0.25, linewidth=0.6)
    axes.legend(loc="upper left", fontsize=8, framealpha=0.9)
    figure.tight_layout()
    for suffix in ("pdf", "png"):
        figure.savefig(PLOTS / f"pm_jax_run11_reward_curves.{suffix}", dpi=180)
    print(f"wrote {PLOTS / 'pm_jax_run11_reward_curves.pdf'} and its .png")


if __name__ == "__main__":
    main()
