"""Draw each arm's best configuration at each of the three learning rates -- six curves.

"Best" here is the **highest point the aggregated curve reaches**: the copies are averaged first,
window by window, and the maximum is taken of that mean. It is deliberately not the mean of each
copy's own best window, which would be larger and would describe no run that ever happened, since
every copy peaks at a different moment. For each (arm, learning rate) the intrinsic weight that
maximises this is the one drawn, so the figure holds one curve per cell of a 2 x 3 grid.

What the figure is for: the learning rate, not the bonus, decides WHEN a run peaks and how far it
falls afterwards. Colour is the arm and line style plus transparency is the learning rate, so the
two arms can be compared at a glance within a rate and across rates within an arm.

The maximum of every curve is marked on the curve itself and labelled there, rather than in the
legend, so a reader never has to match a line to a legend entry to read off its best point.

Run:
  PYTHONNOUSERSITE=1 /p/rlprojects/RND/.venvs/platform_jax/bin/python analysis/code/make_shape_curves.py
"""
import sys
from pathlib import Path

RUN_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(RUN_DIR / "code"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib                                                        # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                          # noqa: E402
import numpy as np                                                       # noqa: E402

from curves_all import curves_all                                        # noqa: E402

PLOTS = RUN_DIR / "analysis" / "plots"
RATES = (1e-3, 1e-4, 1e-5)

# colour is the ARM, and stays the one each arm carries everywhere else in this run family
ARM = {
    "rnd_next_state": ("random network distillation", "black"),
    "gt_position_velocity_sqrt": (r"oracle counts $1/\sqrt{n}$", "#0072B2"),
}
# line style and transparency are the LEARNING RATE: solid and opaque for the rate that wins,
# dotted and faint for the one that peaks earliest, dashed between them
RATE_STYLE = {
    1e-3: ((0, (1.6, 1.8)), 0.62, 1.2),
    1e-4: ("-", 1.00, 1.7),
    1e-5: ((0, (6, 2.6)), 0.80, 1.3),
}
# Where each maximum's label sits, as an offset in points from its marker. The label carries the
# value and the step only: the legend already says which arm and which two hyperparameters the
# curve is, and repeating them here made the label wide enough that the two curves peaking in the
# last window could not be labelled beside their markers at all without running past the spine.
# Set by hand and checked on the render -- the six maxima fall into three close pairs, so the
# members of a pair go opposite ways -- and the check at the end of `main` refuses any label that
# lands outside the axes.
LABEL_OFFSET = {
    ("gt_position_velocity_sqrt", 1e-3): (-58, -6),
    ("rnd_next_state", 1e-3): (-12, 16),
    ("gt_position_velocity_sqrt", 1e-4): (-95, 10),
    ("rnd_next_state", 1e-4): (10, 6),
    ("gt_position_velocity_sqrt", 1e-5): (10, -13),
    ("rnd_next_state", 1e-5): (10, 3),
}


def exponent(value: float) -> int:
    """A swept value's power of ten, for a label; every swept value here is one."""
    return int(round(np.log10(value)))


def best_by_maximum(curves: dict) -> dict:
    """For each (arm, learning rate), the intrinsic weight whose aggregated curve peaks highest.

    before: 66 curves, 11 weights for each of 2 arms x 3 rates;
    after:  6 entries, each the winning weight with its curve and the index of its peak
    """
    best = {}
    for (arm, rate, weight), curve in curves.items():
        means = curve["mean_episode_return"]
        if arm not in ARM or not means:
            continue
        peak = int(np.argmax(means))
        current = best.get((arm, rate))
        if current is None or means[peak] > current["maximum"]:
            best[(arm, rate)] = {"weight": weight, "curve": curve, "peak": peak,
                                 "maximum": means[peak],
                                 "error": curve["standard_error"][peak]}
    return best


def main() -> None:
    """One panel, six curves, each with its maximum marked and labelled on the curve."""
    PLOTS.mkdir(parents=True, exist_ok=True)
    best = best_by_maximum(curves_all(RUN_DIR))
    figure, axes = plt.subplots(1, 1, figsize=(8.0, 5.0))

    for (arm, rate), entry in sorted(best.items(), key=lambda item: (item[0][0], -item[0][1])):
        name, colour = ARM[arm]
        style, alpha, width = RATE_STYLE[rate]
        curve = entry["curve"]
        steps = np.asarray(curve["env_steps_per_copy"], dtype=float) / 1e6
        mean = np.asarray(curve["mean_episode_return"], dtype=float)
        error = np.asarray(curve["standard_error"], dtype=float)
        axes.plot(steps, mean, color=colour, linestyle=style, linewidth=width, alpha=alpha,
                  label=rf"{name}, $\alpha = 10^{{{exponent(rate)}}}$, "
                        rf"$\beta = 10^{{{exponent(entry['weight'])}}}$", zorder=3)
        axes.fill_between(steps, mean - error, mean + error, color=colour,
                          alpha=0.09 * alpha, linewidth=0, zorder=2)

        # the maximum, marked ON the curve and labelled beside the marker
        peak = entry["peak"]
        axes.plot(steps[peak], mean[peak], marker="o", markersize=5.5, color=colour,
                  markeredgecolor="white", markeredgewidth=0.9, alpha=1.0, zorder=5)
        text = rf"${entry['maximum']:.1f}$ at ${steps[peak]:.0f}$M"
        shared = dict(fontsize=8, color=colour, zorder=6,
                      bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none",
                                alpha=0.78))
        axes.annotate(text, xy=(steps[peak], mean[peak]),
                      xytext=LABEL_OFFSET[(arm, rate)], textcoords="offset points", **shared)

    axes.set_xscale("log")
    # room to the right of the last window and above the tallest peak for the maximum labels
    axes.set_xlim(right=5.5e3)
    axes.set_ylim(top=95)
    axes.set_xlabel("environment steps per copy (millions, logarithmic)")
    axes.set_ylabel("mean episode return")
    axes.set_title("Each arm's best configuration at each learning rate, and where it peaks",
                   fontsize=11)
    axes.grid(True, which="major", alpha=0.16, linewidth=0.6)
    axes.legend(fontsize=7.6, loc="upper left", framealpha=0.92)
    figure.tight_layout()
    # every annotation must land inside the axes; a label that runs past the spine is a defect
    figure.canvas.draw()
    box = axes.get_window_extent()
    for child in axes.texts:
        extent = child.get_window_extent()
        if extent.x1 > box.x1 or extent.x0 < box.x0 or extent.y1 > box.y1:
            print(f"  LABEL OUTSIDE AXES: {child.get_text()!r} "
                  f"x=[{extent.x0:.0f},{extent.x1:.0f}] axes x=[{box.x0:.0f},{box.x1:.0f}]")
    print(f"  axes pixel box x=[{box.x0:.0f},{box.x1:.0f}] y=[{box.y0:.0f},{box.y1:.0f}]")
    for suffix in ("pdf", "png"):
        figure.savefig(PLOTS / f"pm_jax_run11_curve_shapes.{suffix}", dpi=190)
    print(f"wrote {PLOTS / 'pm_jax_run11_curve_shapes.pdf'}")
    for (arm, rate), entry in sorted(best.items()):
        print(f"  {arm:28} lr={rate:.0e} beta={entry['weight']:.0e} "
              f"max={entry['maximum']:.3f} +- {entry['error']:.3f}")


if __name__ == "__main__":
    main()
