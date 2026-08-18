"""Draw the development document's training-curve figure after the second extension.

The figure answers the question this run exists to ask: **does the configuration a short budget
picks stay the right one at a long budget?** Both sets of curves run to the same 1000M steps per
copy, so the only difference between them is which configuration was chosen and how:

- **dashed** — the configurations selected from the 10M-step batch, one per arm, as the first
  extension re-ran them at 1000M steps with 1,024 copies each;
- **solid** — the configurations this run's own 1000M-step sweep of the whole grid selects, one per
  arm, over 128 copies each.

Line style therefore encodes which sweep chose the configuration, and colour encodes the arm, so an
arm that appears in both sets is one colour drawn twice. Solid black stays reserved for the best
random-network-distillation configuration, which is the reference every other arm is measured
against.

The figure keeps the file name `pm_jax_run11_reward_curves.pdf` and the document label
`fig:pm-jax-run11-curves`, because it fills the same slot; it is written into THIS run folder, so
the first extension's own plot survives untouched in its folder as the record of what the figure
showed before.

Every configuration is read through `code/aggregate.best_per_arm` rather than hardcoded, so the
figure and the development document's results table cannot name different configurations, and
`aggregate.curve_of` pools a configuration's chunks per copy per window, so a curve is the mean over
all of its copies with the band plus or minus one standard error over them.

**One module reads both runs.** This run's `code/aggregate.py` streams its shards line by line
because they come to about 4 GB; the first extension's own module reads its shards into memory,
which costs about 1.3 GB per call and this figure would make five of them. The streaming module was
checked against the first extension's on that run's own data and reproduces every scored number
exactly (largest relative difference 0), at 15 MB of memory instead of 5 GB, so it is used for both
sets of curves here. The development document's results table still reads each block through the
module of the run that produced it, because that is where the numbers are cited.

Run:
  PYTHONNOUSERSITE=1 /p/rlprojects/RND/.venvs/platform_jax/bin/python analysis/code/make_reward_curves.py
"""
import sys
from pathlib import Path

RUN_DIR = Path(__file__).resolve().parent.parent.parent
RUNS = RUN_DIR.parent
sys.path.insert(0, str(RUN_DIR / "code"))

import matplotlib                                                        # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                          # noqa: E402
import numpy as np                                                       # noqa: E402

from aggregate import best_per_arm, cell_table, curve_of                 # noqa: E402

PLOTS = RUN_DIR / "analysis" / "plots"
FIRST_EXTENSION = RUNS / (
    "2026-08-16-22-47_jax-ppo_pointmaze-large-nonoise_full-batch_rollout-128_envs-per-copy-4"
    "_steps-per-copy-1000M_bonus-rnd-visit-count-sqrt-visit-count-linear-none_learning-rate-1e-3"
    "_intrinsic-weight-rnd-10-visit-count-1_copies-1024_extension-of-first-batch")

# (colour, human name) per arm. Black is the distillation reference and is used by no other arm.
ARM_STYLE = {
    "rnd_next_state": ("black", "random network distillation"),
    "gt_position_velocity_sqrt": ("#0072B2", r"oracle counts $1/\sqrt{n}$"),
    "gt_position_velocity_linear": ("#D55E00", "oracle counts $1/n$"),
    "none": ("#009E73", "no bonus"),
}
ARM_ORDER = ["rnd_next_state", "gt_position_velocity_sqrt", "gt_position_velocity_linear", "none"]


def configuration_note(row: dict) -> str:
    """The learning rate and, where the arm has one, the intrinsic weight, as a legend fragment.

    before: {"learning_rate": 0.001, "intrinsic_weight": 10.0, "bonus": "rnd_next_state"}
    after:  "$\\alpha = 10^{-3}$, $\\beta = 10^{1}$"
    """
    rate = rf"$\alpha = 10^{{{int(round(np.log10(row['learning_rate'])))}}}$"
    if row["bonus"] == "none":
        return rate
    return rf"{rate}, $\beta = 10^{{{int(round(np.log10(row['intrinsic_weight'])))}}}$"


def draw_set(axes, best: dict, curve_source, style: str, chosen_at: str, alpha: float,
             line_alpha: float = 1.0) -> None:
    """Draw one arm-per-curve set in one line style, with its standard-error band.

    before: best = {"rnd_next_state": row, ...}, style = (6, 3);
    after:  one dashed line per arm on `axes`, each labelled with the arm, the configuration and
            how many copies it is a mean over.

    `line_alpha` below 1 is what keeps the two sets apart where the curves are noisy. A dash
    pattern is laid along the DRAWN PATH, so a curve that oscillates within one pixel column spends
    a whole dash period there and reads as a solid line; the reference set is therefore lightened
    as well as dashed.
    """
    for bonus in ARM_ORDER:
        if bonus not in best:
            continue
        row = best[bonus]
        colour, name = ARM_STYLE[bonus]
        curve = curve_source(bonus, row["learning_rate"], row["intrinsic_weight"])
        # before: env_steps_per_copy counted in steps, up to 1,000,038,400;
        # after:  millions, which is the unit this run is planned and reported in
        steps = np.asarray(curve["env_steps_per_copy"], dtype=float) / 1e6
        mean = np.asarray(curve["mean_episode_return"], dtype=float)
        error = np.asarray(curve["standard_error"], dtype=float)
        axes.plot(steps, mean, color=colour, linestyle=style, linewidth=1.2, alpha=line_alpha,
                  label=f"{name}, {chosen_at} ({configuration_note(row)}, n={row['copies']})")
        axes.fill_between(steps, mean - error, mean + error, color=colour, alpha=alpha,
                          linewidth=0)


def main() -> None:
    """Draw one panel: the 10M-chosen configurations dashed, this run's own choices solid."""
    best_here = best_per_arm(cell_table(RUN_DIR, completed_only=False))
    if not best_here:
        raise SystemExit("no scored configuration yet: the run has written no phase-blocked window")
    best_first = best_per_arm(cell_table(FIRST_EXTENSION, completed_only=False))

    PLOTS.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 1, figsize=(7.6, 4.8))
    draw_set(axes, best_first,
             lambda bonus, rate, weight: curve_of(bonus, rate, weight, FIRST_EXTENSION),
             (0, (6, 3)), "chosen at 10M", 0.10, line_alpha=0.55)
    draw_set(axes, best_here,
             lambda bonus, rate, weight: curve_of(bonus, rate, weight, RUN_DIR),
             "-", "chosen at 1000M", 0.18)

    # the step axis is logarithmic: every arm rises to its peak inside the first 15 million steps
    # and then decays over the remaining 985 million, so on a linear axis the whole rise is squeezed
    # into the first per cent of the width and the figure shows a spike against a flat line
    axes.set_xscale("log")
    axes.set_xlabel("environment steps per copy (millions, logarithmic)")
    axes.set_ylabel("mean episode return")
    axes.set_title("PointMaze large, no reset noise — 1000M steps per copy", fontsize=11)
    axes.grid(alpha=0.25, linewidth=0.6, which="both")
    axes.legend(loc="upper left", fontsize=7, framealpha=0.9)
    figure.tight_layout()
    for suffix in ("pdf", "png"):
        figure.savefig(PLOTS / f"pm_jax_run11_reward_curves.{suffix}", dpi=180)
    print(f"wrote {PLOTS / 'pm_jax_run11_reward_curves.pdf'} and its .png")


if __name__ == "__main__":
    main()
