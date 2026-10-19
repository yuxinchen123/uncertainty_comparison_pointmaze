"""Plot mean episode return against environment steps from a summarize_episode_records.py output.

One panel per unit or per intrinsic weight, one line per the other; bands are one standard error
over copies. A terminating task with a -1 step shift is already on the lump-sum scale in the
summary.

Usage: plot_episode_curves.py summary.json out.pdf --panel-by unit|beta [--units a,b,...]
       [--titles a=Title,...] [--reach]   (--reach plots the share of episodes reaching the goal)
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

WEIGHT_COLORS = {0.0: "0.3", 1.0: "tab:orange", 10.0: "tab:green", 100.0: "black",
                 1000.0: "tab:purple"}


def main():
    """Group the summary's settings into panels and lines, then draw them."""
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--panel-by", choices=["unit", "beta"], default="unit")
    parser.add_argument("--units", default="", help="comma-separated units, in panel/line order")
    parser.add_argument("--titles", default="", help="unit=title pairs, comma-separated")
    parser.add_argument("--reach", action="store_true")
    parser.add_argument("--columns", type=int, default=4)
    args = parser.parse_args()
    summary = list(json.loads(args.summary.read_text()).values())
    units = args.units.split(",") if args.units else sorted({s["unit"] for s in summary})
    titles = dict(t.split("=", 1) for t in args.titles.split(",")) if args.titles else {}
    weights = sorted({s["intrinsic_weight"] for s in summary})

    # panels and the lines inside each
    if args.panel_by == "unit":
        panels = [(titles.get(u, u), [s for s in summary if s["unit"] == u]) for u in units]
    else:
        panels = [(f"intrinsic weight {w:g}",
                   sorted([s for s in summary if s["intrinsic_weight"] == w],
                          key=lambda s: units.index(s["unit"]) if s["unit"] in units else 99))
                  for w in weights]
    n = len(panels)
    cols = min(args.columns, n)
    rows = -(-n // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 2.8 * rows), squeeze=False,
                             sharex=True)
    cmap = plt.get_cmap("tab20")
    for ax, (title, members) in zip(axes.flat, panels):
        for k, s in enumerate(members):
            x = np.array(s["curve_steps"]) / 1e6
            if args.reach:
                y, band = np.array(s["reach_share_curve"]), None
            else:
                y, band = np.array(s["curve"]), np.array(s["curve_se"])
            if args.panel_by == "unit":
                color = WEIGHT_COLORS.get(s["intrinsic_weight"], cmap(k))
                label = (f"weight {s['intrinsic_weight']:g}" if s["bonus"] != "none"
                         else "no bonus")
                if s["bonus"] == "gt_position_velocity_sqrt":
                    label = f"oracle 1/sqrt(n), weight {s['intrinsic_weight']:g}"
                    color = "tab:blue"
            else:
                color, label = cmap(k % 20), titles.get(s["unit"], s["unit"])
            ax.plot(x, y, color=color, lw=1.1, label=label)
            if band is not None:
                ax.fill_between(x, y - band, y + band, color=color, alpha=0.2, lw=0)
        ax.set_title(title, fontsize=9)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6, loc="best")
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("environment steps per copy (millions)")
    ylabel = "share of episodes reaching the goal" if args.reach else "mean extrinsic return per episode"
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel, fontsize=8)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)
    fig.savefig(args.output.with_suffix(".png"), dpi=130)


if __name__ == "__main__":
    main()
