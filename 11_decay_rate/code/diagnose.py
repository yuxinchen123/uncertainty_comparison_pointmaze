"""Per-experiment diagnostics (free-standing tool, not part of the fixed protocol): where in
the maze the method fails, and how. Reads one experiment folder's records and writes
diagnostics/report.md + figures into that folder.

Run:  /p/rlprojects/RND/.venvs/exploration/bin/python diagnose.py --exp <experiment folder>
"""
import argparse
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decay_harness.fitting import BURN_IN, fit_power_floor
from decay_harness.metrics import deviation_per_position, target_curve
from decay_harness.points import MAZE_COLS, MAZE_ROWS, point_set

# PointMaze_Large-v3 wall map (True = wall), row 0 at the top — matches the maze used by every
# prior run (46 free cells). Only used to grey wall cells in the heatmaps.
MAZE_MAP = [
    "111111111111",
    "100010000101",
    "101011011101",
    "100000010001",
    "111011110101",
    "100010000101",
    "101110111011",
    "100000000001",
    "111111111111",
]


def load_env_records(exp_dir: str, env: str) -> list:
    """Load the non-diverged records of one point set, sorted by seed."""
    recs = []
    for p in sorted(glob.glob(os.path.join(exp_dir, f"{env}_seed*.json"))):
        with open(p) as fh:
            r = json.load(fh)
        if not r.get("diverged"):
            recs.append(r)
    return recs


def seed_mean_curves(recs: list) -> tuple:
    """(steps (T,), seed-mean bonus (T, P), seed-mean per-position deviation (P,))."""
    steps = np.asarray(recs[0]["checkpoint_steps"])
    b = np.mean([np.asarray(r["bonus"], dtype=float) for r in recs], axis=0)
    dev = np.mean([deviation_per_position(r) for r in recs], axis=0)
    return steps, b, dev


def maze_heatmap(ax, values: np.ndarray, pts: np.ndarray, title: str):
    """Cell-grid heatmap drawn with Rectangles in world coordinates (row 0 top, y up; the
    project's maze plot convention — never imshow, which drops thin walls in vector PDFs)."""
    # map each point's world (x, y) back to its cell (row, col): col = x + 5.5, row = 4 - y
    vmin, vmax = np.nanmin(values), np.nanmax(values)
    cmap = plt.get_cmap("viridis")
    for (x, y), v in zip(pts[:, :2], values):
        col, row = int(round(x + 5.5)), int(round(4.0 - y))
        frac = 0.5 if vmax == vmin else (v - vmin) / (vmax - vmin)
        ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor=cmap(frac), edgecolor="none"))
        if MAZE_MAP[row][col] == "1":
            ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, fill=False,
                                   edgecolor="grey", hatch="///", linewidth=0.4))
    ax.set_xlim(-6, 6); ax.set_ylim(-4.5, 4.5)
    ax.set_aspect("equal"); ax.set_title(f"{title}\n[{vmin:.3f}, {vmax:.3f}] (hatch = wall)",
                                         fontsize=9)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin, vmax))
    plt.colorbar(sm, ax=ax, fraction=0.04)


def main() -> None:
    """Build the report and figures for one experiment folder."""
    ap = argparse.ArgumentParser(description="11_decay_rate experiment diagnostics")
    ap.add_argument("--exp", type=str, required=True)
    args = ap.parse_args()
    out = os.path.join(args.exp, "diagnostics")
    os.makedirs(out, exist_ok=True)
    lines = [f"# Diagnostics — {os.path.basename(args.exp)}", ""]

    for env in ("cell_midpoints", "center_square"):
        recs = load_env_records(args.exp, env)
        if not recs:
            continue
        steps, b, dev = seed_mean_curves(recs)
        pts = point_set(env)
        slopes = np.array([fit_power_floor(steps, b[:, i])["slope"] for i in range(b.shape[1])])
        order = np.argsort(-dev)

        # text report: the ten worst positions with their coordinates, deviation, slope, start
        lines += [f"## {env} ({len(recs)} seeds)", "",
                  "| rank | (x, y) | deviation | fitted slope | b(0) |",
                  "|---|---|---|---|---|"]
        for k in order[:10]:
            lines.append(f"| {list(order).index(k) + 1} | ({pts[k, 0]:+.2f}, {pts[k, 1]:+.2f}) "
                         f"| {dev[k]:.3f} | {slopes[k]:.3f} | {b[0, k]:.3f} |")
        lines += ["", f"slope mean {slopes.mean():.3f}, std {slopes.std():.3f}, "
                      f"min {slopes.min():.3f}, max {slopes.max():.3f}", ""]

        # figure: heatmaps (cell_midpoints only) + worst curves + slope histogram
        n_panels = 4 if env == "cell_midpoints" else 2
        fig, axes = plt.subplots(1, n_panels, figsize=(5.2 * n_panels, 4.4))
        ax_list = list(np.atleast_1d(axes))
        if env == "cell_midpoints":
            maze_heatmap(ax_list.pop(0), dev, pts, "per-position deviation from min(1, n^-1/2)")
            maze_heatmap(ax_list.pop(0), slopes, pts, "per-position fitted slope")
        ax = ax_list.pop(0)
        win = steps >= 1
        tgt = target_curve(np.asarray(recs[0]["visit_counts"], dtype=float))
        for k in order[:6]:
            ax.loglog(steps[win], b[win, k], lw=1.0,
                      label=f"({pts[k, 0]:+.1f}, {pts[k, 1]:+.1f}) dev {dev[k]:.2f}")
        ax.loglog(steps[win], tgt[win, 0], color="grey", lw=2, label="min(1, n^-1/2)")
        ax.set_title("six worst positions (seed-mean)", fontsize=9)
        ax.set_xlabel("step n"); ax.legend(fontsize=6)
        ax = ax_list.pop(0)
        ax.hist(slopes, bins=40)
        ax.axvline(-0.5, color="grey", lw=2)
        ax.set_title("per-position fitted slopes", fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(out, f"{env}.png"), dpi=140)
        plt.close(fig)

    with open(os.path.join(out, "report.md"), "w") as fh:
        fh.write("\n".join(lines))
    print(f"wrote {out}/report.md and figures")


if __name__ == "__main__":
    main()
