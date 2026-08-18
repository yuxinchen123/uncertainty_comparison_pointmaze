"""Draw the two configurations that do NOT rise then fall, against the two that do.

The queued question asked whether every configuration of this sweep rises and then comes down. It
does not, so the answer names the best configuration of each arm that has some other shape — and
neither of those is the arm's own best, so neither appears in the development document's results
table or its curve figure. This is the plot that shows them, and `analysis/shape_classification.md`
is the table.

Each arm's best configuration is drawn dashed beside its non-rise-then-fall one, because the
comparison is the point: the same arm, the same budget, a different weight, a different shape.

Run:
  PYTHONNOUSERSITE=1 /p/rlprojects/RND/.venvs/platform_jax/bin/python analysis/code/make_shape_curves.py
"""
import sys
from pathlib import Path

RUN_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(RUN_DIR / "code"))

import matplotlib                                                        # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                          # noqa: E402
import numpy as np                                                       # noqa: E402

from aggregate import curve_of                                           # noqa: E402

PLOTS = RUN_DIR / "analysis" / "plots"

# (arm, learning rate, intrinsic weight, shape, style) — the arm's best dashed, its best
# non-rise-then-fall solid, one colour per arm as everywhere else in this run family
CURVES = [
    ("rnd_next_state", 1e-4, 1e2, "rise then fall", "black", (0, (6, 3))),
    ("rnd_next_state", 1e-5, 1e1, "still rising", "black", "-"),
    ("gt_position_velocity_sqrt", 1e-4, 1e1, "rise then fall", "#0072B2", (0, (6, 3))),
    ("gt_position_velocity_sqrt", 1e-4, 1e2, "rise then plateau", "#0072B2", "-"),
]
ARM_NAME = {"rnd_next_state": "random network distillation",
            "gt_position_velocity_sqrt": r"oracle counts $1/\sqrt{n}$"}


def main() -> None:
    """One panel: the two shapes each arm produces at 1000M steps, on the same axes."""
    PLOTS.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 1, figsize=(7.6, 4.4))
    for arm, rate, weight, shape, colour, style in CURVES:
        curve = curve_of(arm, rate, weight, RUN_DIR)
        steps = np.asarray(curve["env_steps_per_copy"], dtype=float) / 1e6
        mean = np.asarray(curve["mean_episode_return"], dtype=float)
        error = np.asarray(curve["standard_error"], dtype=float)
        alpha = 0.55 if style != "-" else 1.0
        axes.plot(steps, mean, color=colour, linestyle=style, linewidth=1.2, alpha=alpha,
                  label=rf"{ARM_NAME[arm]}, $\alpha = 10^{{{int(round(np.log10(rate)))}}}$, "
                        rf"$\beta = 10^{{{int(round(np.log10(weight)))}}}$ — {shape}")
        axes.fill_between(steps, mean - error, mean + error, color=colour,
                          alpha=0.10 if style != "-" else 0.18, linewidth=0)
    axes.set_xscale("log")
    axes.set_xlabel("environment steps per copy (millions, logarithmic)")
    axes.set_ylabel("mean episode return")
    axes.set_title("The shapes a 1000M-step budget produces, by arm", fontsize=11)
    axes.grid(alpha=0.25, linewidth=0.6, which="both")
    axes.legend(loc="upper left", fontsize=7.5, framealpha=0.9)
    figure.tight_layout()
    for suffix in ("pdf", "png"):
        figure.savefig(PLOTS / f"pm_jax_run11_curve_shapes.{suffix}", dpi=180)
    print(f"wrote {PLOTS / 'pm_jax_run11_curve_shapes.pdf'} and its .png")


if __name__ == "__main__":
    main()
