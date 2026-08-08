#!/usr/bin/env python
"""The three figures of the non-convergence analysis, all from the one-pass cache.

f1_phenomenon : per-seed success-rate trajectories of the 10M task-R baseline + coverage —
                the find-then-forget picture.
f4_credit     : where the successes happen (steps-to-goal) against the discount horizon, with
                the PointMaze contrast — the credit-assignment picture.
f3_objective  : the intrinsic:extrinsic ratio over training and across the bonus-weight grid —
                the objective-mismatch picture.

Writes each as .pdf and .png into plots/.
"""
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import common as C

OKABE = ["#0072B2", "#009E73", "#E69F00", "#CC79A7", "#56B4E9", "#D55E00"]


def _save(fig, name):
    """Write one figure as pdf+png and close it."""
    os.makedirs(C.PLOTS, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(C.PLOTS, f"{name}.{ext}"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote plots/{name}.pdf/.png")


def f1_phenomenon():
    """2x2: per-seed success-rate spaghetti + median (top), coverage vs success (bottom)."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), sharex="col")
    for col, env in enumerate([C.UMAZE, C.MEDIUM]):
        rows = C.load(source="run12", arm="baseline", env_setup=env, total_timesteps=10_000_000)
        ax = axes[0][col]
        # per-seed curves at 1M-step display bins (20 of the 50k bins each)
        per_seed = [C.bin_success(r, agg=20) for r in rows]
        for steps, rates in per_seed:
            ax.plot([s / 1e6 for s in steps], rates, color="#B0C4DE", linewidth=0.8, alpha=0.7)
        # median across seeds per display bin
        nb = len(per_seed[0][0])
        med = [sorted(pr[1][i] for pr in per_seed)[len(per_seed) // 2] for i in range(nb)]
        ax.plot([s / 1e6 for s in per_seed[0][0]], med, color="black", linewidth=2.2,
                label=f"median (n={len(rows)})")
        ax.set_title(f"{C.ENV_SHORT[env]} — task-R baseline, $10^7$ steps")
        ax.set_ylabel("success rate per 1M-step block" if col == 0 else "")
        ax.legend(fontsize=8)
        # bottom row: median coverage (both grids) with the median success rate re-drawn
        ax2 = axes[1][col]
        n_ev = min(len(r["eval_steps"]) for r in rows)
        ev_steps = rows[0]["eval_steps"][:n_ev]
        for key, style, label in ((("cov_cell"), "-", "maze-cell coverage %"),
                                  (("cov_1m"), "--", "1 m coverage %")):
            med_cov = [sorted(r[key][i] for r in rows if r[key][i] is not None)
                       [len(rows) // 2] for i in range(n_ev)]
            ax2.plot([s / 1e6 for s in ev_steps], med_cov, style, color=OKABE[0], label=label)
        ax2.set_ylabel("coverage %" if col == 0 else "")
        ax2.set_ylim(0, 105)
        axr = ax2.twinx()
        axr.plot([s / 1e6 for s in per_seed[0][0]], med, color="black", linewidth=1.8)
        axr.set_ylabel("success rate (black)" if col == 1 else "")
        ax2.set_xlabel("environment steps (millions)")
        ax2.legend(fontsize=8, loc="center right")
    fig.suptitle("The goal is found, then unlearned, while exploration saturates\n"
                 "(RND baseline at its winning bonus weight; one gray line per seed)")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    _save(fig, "f1_phenomenon")


def f4_credit():
    """Steps-to-goal distributions vs the discount horizon, and the advantage they imply."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    # left: histogram of steps-to-goal of every successful episode, per environment, plus the
    # PointMaze UMaze contrast (run 1.1 RND arm), with the gamma horizon marked
    ax = axes[0]
    cases = [
        ("PointMaze UMaze (cap 300, converges)",
         C.load(source="run11", arm="baseline",
                env_setup="PointMaze_UMaze-v3_start_bottom_left"), 300, OKABE[1]),
        ("AntMaze UMaze (cap 700)",
         C.load(source="run12", arm="baseline", env_setup=C.UMAZE,
                total_timesteps=10_000_000), 700, OKABE[0]),
        ("AntMaze Medium (cap 1000)",
         C.load(source="run12", arm="baseline", env_setup=C.MEDIUM,
                total_timesteps=10_000_000), 1000, OKABE[5]),
    ]
    stats = []
    for label, rows, cap, color in cases:
        lens = [k for r in rows for k in r["succ_lens"]]
        if not lens:
            continue
        ax.hist(lens, bins=40, range=(0, 1000), density=True, histtype="step",
                linewidth=1.8, color=color, label=f"{label}, n={len(lens)}")
        med = sorted(lens)[len(lens) // 2]
        stats.append((label, med, cap, color))
    ax.axvline(1 / (1 - C.GAMMA), color="black", linestyle=":",
               label=r"effective horizon $1/(1-\gamma)=100$")
    ax.set_xlabel("steps to goal of successful episodes")
    ax.set_ylabel("density")
    ax.legend(fontsize=7)
    ax.set_title("Where the successes happen")
    # right: the discounted start-state advantage of succeeding at the observed steps-to-goal
    ax = axes[1]
    for label, med, cap, color in stats:
        adv = C.delta_gamma(med, cap)
        ax.bar(label.split(" (")[0], adv, color=color)
        ax.text(label.split(" (")[0], adv * 1.1, f"{adv:.2f}\n(median $k$={med})",
                ha="center", fontsize=8)
    ax.set_yscale("log")
    ax.set_ylabel(r"$\Delta_\gamma=(\gamma^{k}-\gamma^{T})/(1-\gamma)$")
    ax.axhline(100, color="gray", linestyle="--", linewidth=0.8)
    ax.text(0.02, 102, "return scale ≈ 100", fontsize=8, color="gray")
    ax.set_title("The learning signal a success carries")
    fig.tight_layout()
    _save(fig, "f4_credit")


def f3_objective():
    """The intrinsic:extrinsic ratio: over training (task R) and across the beta grid (1M)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    # left: median per-bin ratio over the 10M task-R runs (reward norm ON), per env
    ax = axes[0]
    for env, color in ((C.UMAZE, OKABE[0]), (C.MEDIUM, OKABE[5])):
        rows = C.load(source="run12", arm="baseline", env_setup=env,
                      total_timesteps=10_000_000)
        nb = len(rows[0]["n_ep"])
        agg = 20  # 1M-step display bins
        steps, meds = [], []
        for i in range(0, nb, agg):
            ratios = []
            for r in rows:
                si = sum(r["sum_int"][i:i + agg])
                se = sum(abs(x) for x in r["sum_ext"][i:i + agg])
                if se > 0:
                    ratios.append(si / se)
            steps.append((i + agg) * C.BIN / 1e6)
            meds.append(sorted(ratios)[len(ratios) // 2] if ratios else float("nan"))
        ax.plot(steps, meds, color=color, marker="o", label=C.ENV_SHORT[env])
    ax.axhline(1.0, color="black", linestyle=":")
    ax.set_xlabel("environment steps (millions)")
    ax.set_ylabel(r"median $\sum|\beta I| \,/\, \sum|E|$ per block")
    ax.set_title("Task-R baseline ($10^7$ steps, reward norm ON):\nthe bonus never anneals away")
    ax.legend(fontsize=8)
    # right: the same ratio at 1M across the beta grid for the reward-norm-OFF task-S arms
    ax = axes[1]
    rows = C.load(source="run12", arm=["alg1", "alg2.1", "alg2.2", "alg2.3"],
                  env_setup=C.UMAZE, total_timesteps=1_000_000)
    by_beta = {}
    for r in rows:
        si, se = sum(r["sum_int"]), sum(abs(x) for x in r["sum_ext"])
        if se > 0:
            by_beta.setdefault(float(r["beta"]), []).append(si / se)
    betas = sorted(by_beta)
    meds = [sorted(by_beta[b])[len(by_beta[b]) // 2] for b in betas]
    ax.plot(betas, meds, marker="o", color=OKABE[2], label="all four algorithm arms pooled")
    ax.axhline(1.0, color="black", linestyle=":")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"bonus weight $\beta$")
    ax.set_ylabel(r"median episode $|\beta I|/|E|$")
    ax.set_title("Task-S arms at $10^6$ steps (reward norm OFF):\nthe winning weights sit far above 1")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, "f3_objective")


if __name__ == "__main__":
    f1_phenomenon()
    f4_credit()
    f3_objective()
