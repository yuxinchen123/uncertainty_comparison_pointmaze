#!/usr/bin/env python
"""Controlled test: does a nonzero bias initialization flatten the untrained RND bonus bowl?

Question: the untrained RND network's bonus B(x) = 0.5*||predictor(x) - target(x)||^2 starts as a
"bowl" over the maze — near zero at the whitened center and growing outward (the reference heatmap
figure shows a max/min range of thousands). If that bowl is only an artifact of both nets having
their biases initialized to exactly 0, then giving the biases a nonzero starting value should raise
the floor and flatten the bowl. This script measures that directly.

What is held fixed (the control): for one seed, the network weights and the observation normalizer
are built once by the reference script's build_untrained_rnd(a_seed) and REUSED for every bias
scheme. Only the four bias tensors (target + predictor, each net's two Linear layers) are overwritten
in place. So any change in the bonus field across schemes is due to the biases alone, nothing else.

Bias schemes (applied to BOTH nets, BOTH Linear layers):
- zero            : biases exactly 0 (the baseline; must reproduce the reference field bit-for-bit).
- pytorch_default : per-layer U(-1/sqrt(fan_in), +1/sqrt(fan_in)); fan_in = 4 (layer 1) or 256 (layer 2).
- normal_0.1 .. normal_2.0 : b ~ N(0, sigma^2) with sigma in {0.1, 0.25, 0.5, 1.0, 2.0}.

The normal draws are keyed WITHOUT the sigma value (per the reproducible-seeding rule), so the five
scales all scale the SAME underlying standard-normal direction — a pure scale sweep.

Reuses the reference script (build_untrained_rnd, grid_eval, bonus_at, wall_rects, cell_center, the
geometry constants) by importing it with importlib, so none of its ~250 lines are copied.

Outputs (this folder): metrics.json, metrics_table.md, heatmaps_by_bias_scheme.{pdf,png},
bonus_vs_radius.{pdf,png}.
Run:  conda run -n exploration python make_bias_ablation.py
"""
import hashlib
import importlib.util
import json
import os

import numpy as np
import torch
from scipy.stats import spearmanr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
# import the reference untrained-RND script from its sibling folder (no code copied)
REF_PATH = os.path.join(HERE, "..", "2026-07-09-23-43_rnd-init-bonus-field-surface-heatmap",
                        "make_plots.py")
_spec = importlib.util.spec_from_file_location("ref_untrained_rnd", os.path.abspath(REF_PATH))
ref = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ref)

# geometry / evaluation constants pulled straight from the reference so the two agree exactly
LEFT, RIGHT, BOTTOM, TOP = ref.LEFT, ref.RIGHT, ref.BOTTOM, ref.TOP
V_MAX = ref.V_MAX                                  # env velocity clip (+-5)
N_DENSE = ref.N_DENSE                              # 100 x 100 dense grid, same as the reference
VEL_LEVELS = [0.0, V_MAX / 2, V_MAX]              # 0, 2.5, 5 for the global-ratio / velocity metrics

SCHEMES = ["zero", "pytorch_default",
           "normal_0.1", "normal_0.25", "normal_0.5", "normal_1.0", "normal_2.0"]
SEEDS = [111, 0, 1, 2, 3]                          # 111 first (the seed of the writeup figures)
FIG_SEED = 111                                     # both figures are drawn from this seed


def keyed_gen(*parts) -> torch.Generator:
    """One torch.Generator seeded by a hash of stable names (reproducible-seeding rule)."""
    # e.g. keyed_gen(111, 'bias-normal', 'target', 0) -> a fixed generator independent of any sigma
    key = "::".join(str(p) for p in parts)
    g = torch.Generator()
    g.manual_seed(int(hashlib.sha256(key.encode()).hexdigest(), 16) & 0x7FFFFFFF)
    return g


def linears(model):
    """The four bias-bearing Linear layers as (net_name, layer_index, module) tuples."""
    # standard RND: each net is Sequential(Linear(4,256), ReLU, Linear(256,128)); Linears at index 0, 2
    return [("target", 0, model.target.network[0]), ("target", 2, model.target.network[2]),
            ("predictor", 0, model.predictor.network[0]), ("predictor", 2, model.predictor.network[2])]


def set_biases(model, scheme: str, a_seed: int) -> None:
    """Overwrite both nets' two Linear biases in place for one bias scheme; weights are never touched."""
    # walk the four Linear layers; each gets its own keyed draw so weights stay byte-identical
    for net_name, layer_idx, layer in linears(model):
        bias = layer.bias
        fan_in = layer.in_features                 # 4 for layer 0, 256 for layer 2
        with torch.no_grad():
            if scheme == "zero":
                # baseline: exactly 0, reproducing the reference field
                bias.zero_()
            elif scheme == "pytorch_default":
                # PyTorch Linear default bias: U(-1/sqrt(fan_in), +1/sqrt(fan_in)).
                # before: u ~ U(-1, 1), shape (fan_out,); after: bias = u / sqrt(fan_in)
                #   e.g. fan_in=4 -> entries in [-0.5, 0.5]; fan_in=256 -> entries in [-0.0625, 0.0625]
                g = keyed_gen(a_seed, "bias-uniform", net_name, layer_idx)
                u = torch.rand(bias.shape, generator=g) * 2.0 - 1.0
                bias.copy_(u / (fan_in ** 0.5))
            else:
                # normal_<sigma>: b = sigma * z, z ~ N(0,1). z is keyed WITHOUT sigma, so every scale
                # reuses the SAME z direction.  before: z fixed per (net, layer); after: sigma * z
                sigma = float(scheme.split("_")[1])
                g = keyed_gen(a_seed, "bias-normal", net_name, layer_idx)
                z = torch.randn(bias.shape, generator=g)
                bias.copy_(sigma * z)


def free_mask(maze_map, x_centers, y_centers):
    """Boolean (n_y, n_x) grid: True where the grid point sits in an OPEN maze cell (not a wall)."""
    # map each grid point (x, y) to its maze cell (row 0 = TOP, matching wall_rects), then read the map
    # before: x_centers, y_centers of length n; after: (n_y, n_x) mask, ~0.43 True for this maze
    rows, cols = maze_map.shape
    cell_w, cell_h = (RIGHT - LEFT) / cols, (TOP - BOTTOM) / rows
    XX, YY = np.meshgrid(x_centers, y_centers)                       # (n_y, n_x)
    col_idx = np.clip(((XX - LEFT) / cell_w).astype(int), 0, cols - 1)
    row_idx = np.clip(((TOP - YY) / cell_h).astype(int), 0, rows - 1)
    return maze_map[row_idx, col_idx] != 1


def bonus_obs(model, obs):
    """Raw RND bonus B = 0.5*||predictor - target||^2 for an explicit (N, 4) observation array."""
    with torch.no_grad():
        out = model.compute({"next_observations": np.asarray(obs, dtype=np.float32)})
    return np.asarray(out, dtype=float)


def whitened_norms(model, x_centers, y_centers, v):
    """L2 norm ||xbar|| of each grid point's WHITENED RND input (obs_rms whiten + clip +-5) at velocity v."""
    # the whitened input is exactly what compute() feeds the nets; its norm measures distance to center
    # before: [x, y, v, v] raw; after: xbar = clip((obs-mean)/sqrt(var+1e-8), -5, 5), then its L2 norm
    XX, YY = np.meshgrid(x_centers, y_centers)
    obs = np.stack([XX.ravel(), YY.ravel(),
                    np.full(XX.size, v), np.full(XX.size, v)], axis=1).astype(np.float32)
    with torch.no_grad():
        xbar = model._normalize_obs(torch.from_numpy(obs)).numpy()
    return np.linalg.norm(xbar, axis=1).reshape(XX.shape)


def scheme_metrics(model, maze_map, start_cell, x_centers, y_centers, mask):
    """All seven per-(scheme, seed) metrics of the bonus field, computed over FREE maze points."""
    # three velocity grids (v=0, 2.5, 5); the v=0 field drives the shape metrics
    grids = {v: ref.grid_eval(model, N_DENSE, v)[2] for v in VEL_LEVELS}
    v0 = grids[0.0]
    free_v0 = v0[mask]                                              # 1-D bonus over free points

    # bowl shape at v=0: raw range, its ratio, and the coefficient of variation std/mean
    raw_min, raw_max = float(free_v0.min()), float(free_v0.max())
    ratio = raw_max / raw_min
    cv = float(free_v0.std(ddof=0) / free_v0.mean())               # CV = std / mean over free points

    # center bonus: the free point whose whitened input norm ||xbar|| is smallest (the bowl's bottom)
    norms = whitened_norms(model, x_centers, y_centers, 0.0)
    free_norms = norms[mask]
    i_center = int(np.argmin(free_norms))
    center_bonus = float(free_v0[i_center])
    min_norm = float(free_norms[i_center])

    # Spearman rank correlation between the bonus and the whitened radius over free points
    # (near +1 means the bonus rises monotonically with distance from the center -> a clean bowl)
    spearman_r = float(spearmanr(free_v0, free_norms)[0])

    # velocity ratio at the start position: bonus at (vx=5, vy=0) over bonus at (0, 0)
    sx, sy = ref.cell_center(start_cell[0], start_cell[1], maze_map)
    b_v0 = float(bonus_obs(model, [[sx, sy, 0.0, 0.0]])[0])
    b_vx5 = float(bonus_obs(model, [[sx, sy, V_MAX, 0.0]])[0])
    vel_ratio = b_vx5 / b_v0

    # global ratio (shared-normalization dynamic range, as in the writeup): the largest free-point
    # bonus over v in {0, 2.5, 5}, divided by the smallest free-point bonus at v=0
    global_max = max(float(grids[v][mask].max()) for v in VEL_LEVELS)
    global_ratio = global_max / raw_min

    return {"raw_min_v0": raw_min, "raw_max_v0": raw_max, "ratio_v0": ratio, "cv_v0": cv,
            "center_bonus": center_bonus, "min_whitened_norm": min_norm,
            "spearman_r_norm": spearman_r, "vel_ratio": vel_ratio, "global_ratio": global_ratio,
            "_v0_field": v0}


def summarize(per_scheme_seed):
    """Per-scheme mean and sample sd (ddof=1) across seeds for each numeric metric."""
    # collect each metric across the seeds, then reduce; before: {scheme:{seed:{metric:val}}}
    # after: {scheme:{metric:{'mean':m,'sd':s}}}
    keys = ["raw_min_v0", "raw_max_v0", "ratio_v0", "cv_v0", "center_bonus",
            "min_whitened_norm", "spearman_r_norm", "vel_ratio", "global_ratio"]
    out = {}
    for scheme, by_seed in per_scheme_seed.items():
        out[scheme] = {}
        for k in keys:
            vals = np.array([by_seed[str(s)][k] for s in SEEDS], dtype=float)
            out[scheme][k] = {"mean": float(vals.mean()),
                              "sd": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0}
    return out


def fnum(x: float) -> str:
    """Compact number formatting that stays readable from ~1e-3 up to thousands."""
    if x == 0:
        return "0"
    ax = abs(x)
    if ax >= 1000:
        return f"{x:,.0f}"
    if ax >= 100:
        return f"{x:.0f}"
    if ax >= 1:
        return f"{x:.2f}"
    if ax >= 0.01:
        return f"{x:.3f}"
    return f"{x:.2e}"


def write_table(summary, path):
    """Write metrics_table.md: schemes as rows, metrics as columns; smallest ratio bold, 2nd underlined."""
    # rank schemes by mean ratio at v=0 (smaller = flatter bowl = better) to mark best / second-best
    ratios = {s: summary[s]["ratio_v0"]["mean"] for s in SCHEMES}
    order = sorted(SCHEMES, key=lambda s: ratios[s])
    best, second = order[0], order[1]

    # header (narrow cells; long labels wrapped with <br> per the markdown-table-width rule)
    lines = [
        "# Bias initialization ablation on the untrained RND bonus field",
        "",
        "One row per bias scheme. All values are means across seeds "
        f"{SEEDS} except **ratio v0**, which shows mean ± sd. Metrics are computed over the free "
        "(non-wall) points of a 100×100 grid at velocity 0 unless noted.",
        "",
        "Definitions (bonus $B=\\tfrac12\\lVert \\text{predictor}-\\text{target}\\rVert^2$, over free "
        "points): ratio v0 $=B_{\\max}/B_{\\min}$; center bonus $=B$ at the free point with the "
        "smallest whitened input norm $\\lVert\\bar x\\rVert$; max bonus v0 $=B_{\\max}$; "
        "Spearman r = rank correlation of $B$ with $\\lVert\\bar x\\rVert$; "
        "coeff. of variation $=\\operatorname{std}(B)/\\operatorname{mean}(B)$; "
        "velocity ratio $=B(v_x{=}5)/B(v_x{=}0)$ at the start cell; "
        "global ratio $=\\max_{v\\in\\{0,2.5,5\\}}B \\,/\\, \\min_{v=0}B$.",
        "",
        "Smaller ratio v0 = flatter field. **Bold** = smallest ratio v0, <u>underline</u> = second smallest.",
        "",
        "| bias scheme | ratio v0<br>(mean ± sd) | center<br>bonus | max bonus<br>v0 | "
        "Spearman r<br>(B vs radius) | coeff. of<br>variation | velocity<br>ratio | global<br>ratio |",
        "|---|---|---|---|---|---|---|---|",
    ]
    # one row per scheme, in the fixed scheme order
    for s in SCHEMES:
        m = summary[s]
        ratio_cell = f"{fnum(m['ratio_v0']['mean'])} ± {fnum(m['ratio_v0']['sd'])}"
        if s == best:
            ratio_cell = f"**{ratio_cell}**"
        elif s == second:
            ratio_cell = f"<u>{ratio_cell}</u>"
        lines.append(
            f"| {s} | {ratio_cell} | {fnum(m['center_bonus']['mean'])} | "
            f"{fnum(m['raw_max_v0']['mean'])} | {m['spearman_r_norm']['mean']:.3f} | "
            f"{fnum(m['cv_v0']['mean'])} | {fnum(m['vel_ratio']['mean'])} | "
            f"{fnum(m['global_ratio']['mean'])} |")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def fig_heatmaps(fig_fields, maze_map, x_centers, y_centers, mask, path):
    """Figure A: a row of 7 self-normalized (each panel / its own free-space max) v=0 heatmaps, seed 111."""
    n = len(x_centers)
    x_edges = np.linspace(LEFT, RIGHT, n + 1)
    y_edges = np.linspace(BOTTOM, TOP, n + 1)
    cmap = plt.get_cmap("jet").copy()
    cmap.set_bad("white")                                          # wall cells render blank, then overlaid
    fig, axes = plt.subplots(1, len(SCHEMES), figsize=(3.05 * len(SCHEMES), 4.0))
    pm = None
    for ax, scheme in zip(axes, SCHEMES):
        d = fig_fields[scheme]
        # self-normalize by this panel's own free-space max; blank out wall cells so only free color shows
        # before: raw field (n,n); after: field / free_max with wall cells set to NaN (drawn blank)
        disp = d["_v0_field"] / d["raw_max_v0"]
        disp = np.where(mask, disp, np.nan)
        pm = ax.pcolormesh(x_edges, y_edges, disp, cmap=cmap, vmin=0.0, vmax=1.0)
        # true maze walls as solid gray rectangles (vector patches, never imshow)
        for (x, y, w, h) in ref.wall_rects(maze_map):
            ax.add_patch(Rectangle((x, y), w, h, facecolor="#3c3c3c", edgecolor="none"))
        ax.set_xlim(LEFT, RIGHT); ax.set_ylim(BOTTOM, TOP)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(scheme, fontsize=12)
        # raw range + ratio annotated under the panel
        ax.set_xlabel(f"raw B: {fnum(d['raw_min_v0'])} .. {fnum(d['raw_max_v0'])}\n"
                      f"ratio = {fnum(d['ratio_v0'])}", fontsize=10)
    fig.suptitle("Untrained RND bonus at velocity 0, self-normalized per panel (seed 111) — "
                 "each panel divided by its own free-space maximum", fontsize=13, y=1.02)
    cb = fig.colorbar(pm, ax=axes, fraction=0.012, pad=0.01)
    cb.set_label("bonus / panel free-space max")
    fig.savefig(path + ".pdf", bbox_inches="tight")
    fig.savefig(path + ".png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_bonus_vs_radius(fig_fields, radius_free, path):
    """Figure B: median raw bonus vs whitened radius (v=0 free points, ~20 bins), one line per scheme."""
    # shared bins: the whitened radius is identical across schemes (same normalizer), only B differs
    # before: radius_free (F,), bonus_free (F,); after: median bonus per radius bin, per scheme
    n_bins = 20
    edges = np.linspace(radius_free.min(), radius_free.max(), n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    which = np.clip(np.digitize(radius_free, edges) - 1, 0, n_bins - 1)
    colors = ["#000000", "#7f7f7f", "#1f77b4", "#17becf", "#ff7f0e", "#d62728", "#9467bd"]

    fig, ax = plt.subplots(figsize=(9.2, 6.0))
    for scheme, color in zip(SCHEMES, colors):
        bonus_free = fig_fields[scheme]["_bonus_free"]
        # median bonus in each radius bin (empty bins -> NaN, dropped from the line)
        med = np.array([np.median(bonus_free[which == b]) if np.any(which == b) else np.nan
                        for b in range(n_bins)])
        ok = ~np.isnan(med)
        ax.plot(centers[ok], med[ok], marker="o", ms=4, lw=2.2, color=color, label=scheme)
    ax.set_yscale("log")
    ax.set_xlabel(r"whitened input radius $\Vert\bar x\Vert$  (distance from the field's center)",
                  fontsize=13)
    ax.set_ylabel(r"median raw RND bonus  $\frac{1}{2}\Vert e\Vert^2$", fontsize=13)
    ax.set_title("Untrained RND bonus vs distance from center, by bias scheme (seed 111, velocity 0)",
                 fontsize=13)
    ax.legend(title="bias scheme", fontsize=11, frameon=False)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(path + ".pdf", bbox_inches="tight")
    fig.savefig(path + ".png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Run every (scheme, seed), assert the sanity checks, write metrics + both figures."""
    per_scheme_seed = {s: {} for s in SCHEMES}
    fig_fields = {}                                                # seed-111 fields for the figures
    fig_radius = None
    fig_geom = {}                                                  # seed-111 maze/grid/mask for figure A

    for seed in SEEDS:
        # build ONE untrained model per seed (weights + obs_rms) and reuse it for every scheme
        model, maze_map, goal_cell, start_cell, ctx = ref.build_untrained_rnd(seed)
        x_centers, y_centers, _ = ref.grid_eval(model, N_DENSE, 0.0)
        mask = free_mask(maze_map, x_centers, y_centers)
        # independent reference build for the zero-scheme reproduction check
        ref_model = ref.build_untrained_rnd(seed)[0]
        ref_v0 = ref.grid_eval(ref_model, N_DENSE, 0.0)[2]
        # snapshot the four weight tensors to prove bias overwrites never touch the weights
        baseline_weights = {(nm, li): lay.weight.detach().clone() for nm, li, lay in linears(model)}

        for scheme in SCHEMES:
            set_biases(model, scheme, seed)

            # sanity (b): weights unchanged after the bias overwrite
            for nm, li, lay in linears(model):
                assert torch.equal(lay.weight, baseline_weights[(nm, li)]), \
                    f"weight moved: {nm} layer {li}, seed {seed}, scheme {scheme}"
            # sanity (d): target parameters (weights + biases) are frozen
            assert all(not p.requires_grad for p in model.target.parameters()), \
                f"target params not frozen (seed {seed}, scheme {scheme})"
            # sanity (c): normal_* realized std within 20% of sigma (768 pooled bias entries)
            if scheme.startswith("normal_"):
                sigma = float(scheme.split("_")[1])
                pooled = torch.cat([lay.bias.detach().flatten() for _, _, lay in linears(model)])
                realized = float(pooled.std(unbiased=False))
                assert 0.8 * sigma <= realized <= 1.2 * sigma, \
                    f"realized bias std {realized:.3f} off sigma {sigma} (seed {seed})"

            m = scheme_metrics(model, maze_map, start_cell, x_centers, y_centers, mask)

            # sanity (a): the zero scheme reproduces the independent reference field bit-for-bit
            if scheme == "zero":
                assert np.array_equal(m["_v0_field"], ref_v0), \
                    f"zero scheme does not reproduce reference field (seed {seed})"

            # stash seed-111 fields for the figures (v=0 field, free-point bonus, whitened radius)
            if seed == FIG_SEED:
                free_v0 = m["_v0_field"][mask]
                fig_fields[scheme] = {"_v0_field": m["_v0_field"], "_bonus_free": free_v0,
                                      "raw_min_v0": m["raw_min_v0"], "raw_max_v0": m["raw_max_v0"],
                                      "ratio_v0": m["ratio_v0"]}
                if fig_radius is None:
                    fig_radius = whitened_norms(model, x_centers, y_centers, 0.0)[mask]
                    fig_geom = {"maze_map": maze_map, "x_centers": x_centers,
                                "y_centers": y_centers, "mask": mask}

            m.pop("_v0_field")                                     # drop the array before JSON dump
            per_scheme_seed[scheme][str(seed)] = m
            print(f"[bias] seed {seed:3d} {scheme:15s}: ratio_v0={m['ratio_v0']:.1f} "
                  f"center={m['center_bonus']:.3g} max={m['raw_max_v0']:.1f} "
                  f"spearman={m['spearman_r_norm']:.3f} vel={m['vel_ratio']:.2f}")

    # aggregate across seeds and write the JSON + table
    summary = summarize(per_scheme_seed)
    with open(os.path.join(HERE, "metrics.json"), "w") as fh:
        json.dump({"schemes": SCHEMES, "seeds": SEEDS, "n_dense": N_DENSE,
                   "velocity_levels": VEL_LEVELS,
                   "per_scheme_seed": per_scheme_seed, "per_scheme_summary": summary},
                  fh, indent=1)
    write_table(summary, os.path.join(HERE, "metrics_table.md"))

    # both figures from the seed-111 fields (geometry captured during the seed-111 loop iteration)
    fig_heatmaps(fig_fields, fig_geom["maze_map"], fig_geom["x_centers"], fig_geom["y_centers"],
                 fig_geom["mask"], os.path.join(HERE, "heatmaps_by_bias_scheme"))
    fig_bonus_vs_radius(fig_fields, fig_radius, os.path.join(HERE, "bonus_vs_radius"))

    # headline: seed-111 numbers for the console
    print("\n[bias] seed-111 headline (ratio / center bonus / spearman):")
    for s in SCHEMES:
        r = per_scheme_seed[s]["111"]
        print(f"    {s:15s}: ratio={r['ratio_v0']:.1f}  center={r['center_bonus']:.4g}  "
              f"spearman={r['spearman_r_norm']:.3f}")
    print(f"\n[bias] wrote metrics.json, metrics_table.md, 2 figures (pdf+png) under {HERE}")


if __name__ == "__main__":
    main()
