"""Generate the unified report (report.md + figures/) from the benchmark result JSONs.

Rerunnable: reads the NEWEST JSON per (kind, tag) from benchmarks/results/ plus the final
training-run records under train_runs/, writes figures and splices every table into
report.md. Sections whose data has not landed yet render a PENDING marker and the script
prints a warning for each.

Run: <python with matplotlib> make_report.py
"""
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REPORT = HERE.parent
BASE = REPORT.parent.parent
RESULTS = BASE / "benchmarks" / "results"
FIGS = REPORT / "figures"

# fixed categorical order (validated default palette, light mode): color follows the
# ENTITY (framework); line style separates modes of the same entity
C_TORCH = "#2a78d6"   # slot 1 blue
C_CUDA = "#eb6834"    # slot 2 orange
C_JAX = "#1baf7a"     # slot 3 aqua
GRID = dict(color="#d9d9d9", linewidth=0.6)
PENDING = []


def style_ax(ax):
    """Recessive grid and spines per the chart-design rules."""
    ax.grid(True, which="major", **GRID)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def newest(pattern):
    """Newest results JSON whose filename matches the regex, or None."""
    hits = sorted(p for p in RESULTS.glob("*.json") if re.search(pattern, p.name))
    return json.loads(hits[-1].read_text()) if hits else None


def pending(section, what):
    """Record a missing-data section and return its placeholder line."""
    PENDING.append(f"{section}: waiting on {what}")
    return f"*PENDING — waiting on {what}.*\n"


def fig_env_throughput():
    """Figure 1+2: env throughput and per-batch step time vs n_envs (log axes)."""
    series = []
    t = newest(r"envbench_torch_compile_grid")
    if t:
        series.append(("torch (compiled)", C_TORCH, "-", "o", t["rows"]))
    c = newest(r"envbench_cuda.*_grid")
    if c:
        series.append(("CUDA kernel", C_CUDA, "-", "s", c["rows"]))
    j = newest(r"envbench_jax_jit_grid")
    if j:
        series.append(("jax (jit per step)", C_JAX, "-", "^", j["rows"]))
    js = newest(r"envbench_jax_scan_grid")
    if js:
        series.append(("jax (scan, fused rollout)", C_JAX, "--", "v", js["rows"]))
    if not series:
        return pending("Module 1 figures", "env grid JSONs")

    for fname, ykey, ylabel, title in [
        ("env_throughput.png", "env_steps_per_sec", "environment steps / second",
         "Batched PointMaze throughput on one H100"),
        ("env_steptime.png", "us_per_batch_step", "time per batched step [us]",
         "Per-step wall time vs number of environments"),
    ]:
        fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=160)
        for label, color, ls, mk, rows in series:
            xs = [r["total_envs"] for r in rows]
            ys = [r[ykey] for r in rows]
            ax.plot(xs, ys, ls, color=color, linewidth=2, marker=mk, markersize=6,
                    label=label)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("number of parallel environments (log scale)")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11)
        ax.legend(frameon=False, fontsize=9)
        style_ax(ax)
        fig.tight_layout()
        fig.savefig(FIGS / fname)
        plt.close(fig)
    return None


def fig_copy_scaling():
    """Figure 3: total and per-copy training throughput vs copy count (torch + jax)."""
    rows_t0 = (newest(r"trainbench_torch_epoch_minibatch_onegraph\.") or {}).get("rows", [])
    rows_t = (newest(r"trainbench_torch_epoch_minibatch_cscale\.") or {}).get("rows", [])
    rows_t2 = (newest(r"trainbench_torch_epoch_minibatch_cscale2") or {}).get("rows", [])
    rows_j = (newest(r"trainbench_jax_ppo_cscale") or {}).get("rows", [])
    seen = {r["n_copies"] for r in rows_t + rows_t2}
    allt = sorted([r for r in rows_t0 if r["n_copies"] not in seen] + rows_t + rows_t2,
                  key=lambda r: r["n_copies"])
    if not allt:
        return pending("copy-scaling figures", "trainbench cscale JSONs")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), dpi=160)
    panels = [("env_steps_per_sec", "TOTAL env steps / second (training)"),
              ("env_steps_per_sec_per_copy", "per-copy env steps / second")]
    for ax, (key, ylabel) in zip(axes, panels):
        xs = [r["n_copies"] for r in allt]
        ys = [r[key] for r in allt]
        ax.plot(xs, ys, "-", color=C_TORCH, linewidth=2, marker="o", markersize=6,
                label="torch (one-graph, style B)")
        if rows_j:
            xj = [r["n_copies"] for r in rows_j]
            yj = [r.get(key) or r["env_steps_per_sec"] / r["n_copies"] for r in rows_j]
            ax.plot(xj, yj, "-", color=C_JAX, linewidth=2, marker="^", markersize=6,
                    label="jax (style B)")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("independent training copies C (log scale)")
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False, fontsize=9)
        style_ax(ax)
    fig.suptitle("Training throughput vs number of independent copies (T=128, N=4)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGS / "copy_scaling.png")
    plt.close(fig)
    return None


def main():
    FIGS.mkdir(exist_ok=True)
    notes = [fig_env_throughput(), fig_copy_scaling()]
    print("figures written to", FIGS)
    for p in PENDING:
        print("PENDING:", p)


if __name__ == "__main__":
    main()
