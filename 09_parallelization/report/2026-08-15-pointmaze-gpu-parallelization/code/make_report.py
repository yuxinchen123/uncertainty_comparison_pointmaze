"""Generate the unified report (report.md + figures/) from the benchmark result JSONs.

Rerunnable: reads the NEWEST JSON per (kind, tag) from benchmarks/results/ plus the final
training-run records under train_runs/, writes figures and every table into report.md.
Sections whose data has not landed yet render a PENDING marker and the script prints a
warning for each.

Run: <python with matplotlib> make_report.py
"""
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))

HERE = Path(__file__).resolve().parent
REPORT = HERE.parent
BASE = REPORT.parent.parent
RESULTS = BASE / "benchmarks" / "results"
RUNS = BASE / "train_runs"
FIGS = REPORT / "figures"

# the standalone report's processor module: its two sections and the unit conversions they use are
# defined once, there, and reused here so the two documents cannot come to disagree
sys.path.insert(0, str(BASE / "report"
                       / "2026-08-15-gpu-parallel-rl-environment-training-endtoend" / "code"))
import cpu_sections
from cpu_sections import H

# fixed categorical order (validated default palette, light mode): color follows the
# ENTITY (framework); line style separates modes of the same entity
C_TORCH = "#2a78d6"   # slot 1 blue
C_CUDA = "#eb6834"    # slot 2 orange
C_JAX = "#1baf7a"     # slot 3 aqua
C_EXTRA = "#eda100"   # slot 4 yellow (copy counts in the campaign figure use a blue ramp instead)
GRID = dict(color="#d9d9d9", linewidth=0.6)
PENDING = []
STEPS_PER_COPY_PER_ITER = 128 * 4   # num_steps x n_envs: one iteration's steps for one copy


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
    return f"\n*PENDING — waiting on {what}.*\n"


def sci(x):
    """Scientific-notation cell, e.g. 4.03e9."""
    return f"{x:.2e}".replace("e+0", "e").replace("e+", "e")


# ---------------- figures ----------------

def fig_env_throughput():
    """Env throughput and per-batch step time vs n_envs (log axes)."""
    series = []
    t = newest(r"envbench_torch_compile_grid")
    if t:
        series.append(("torch (compiled)", C_TORCH, "-", "o", t["rows"]))
    c = newest(r"envbench_cuda_fused_tourn_count") or newest(r"envbench_cuda.*_grid")
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


def torch_cscale_rows(round2=True):
    """The torch copy-scaling series, one round at a time.

    round2: the shipped configuration measured after round two (C=8..512, and the larger
    counts once they are re-measured). round1: the round-one configuration, which is where
    the memory ceiling at 32,768 copies was found.
    """
    if round2:
        rows = (newest(r"trainbench_torch_epoch_minibatch_round2c_large") or {}).get("rows", [])
        base = (newest(r"trainbench_torch_epoch_minibatch_round2b_styleB") or {}).get("rows", [])
        seen = {r["n_copies"] for r in rows}
        return sorted(base + [r for r in rows if r["n_copies"] not in seen],
                      key=lambda r: r["n_copies"])
    rows_t0 = (newest(r"trainbench_torch_epoch_minibatch_pack\.") or
               newest(r"trainbench_torch_epoch_minibatch_onegraph\.") or {}).get("rows", [])
    rows_t = (newest(r"trainbench_torch_epoch_minibatch_cscale\.") or {}).get("rows", [])
    rows_t2 = (newest(r"trainbench_torch_epoch_minibatch_cscale2") or {}).get("rows", [])
    seen = {r["n_copies"] for r in rows_t + rows_t2}
    return sorted([r for r in rows_t0 if r["n_copies"] not in seen] + rows_t + rows_t2,
                  key=lambda r: r["n_copies"])


def fig_copy_scaling():
    """Total and per-copy training throughput vs copy count (torch + jax)."""
    allt = torch_cscale_rows(round2=True)
    r1 = torch_cscale_rows(round2=False)
    rows_j = (newest(r"trainbench_jax_ppo_cscale") or {}).get("rows", [])
    rows_jb = [r for r in rows_j if r.get("style", "epoch_minibatch") == "epoch_minibatch"]
    if not allt:
        return pending("copy-scaling figures", "trainbench cscale JSONs")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), dpi=160)
    panels = [("env_steps_per_sec", "TOTAL env steps / second (training)"),
              ("env_steps_per_sec_per_copy", "per-copy env steps / second")]
    for ax, (key, ylabel) in zip(axes, panels):
        xs = [r["n_copies"] for r in allt]
        ys = [r[key] for r in allt]
        ax.plot([r["n_copies"] for r in r1], [r[key] for r in r1], "--", color=C_TORCH,
                linewidth=1.6, marker="o", markersize=5, alpha=0.55,
                label="torch, round 1 (style B)")
        ax.plot(xs, ys, "-", color=C_TORCH, linewidth=2, marker="o", markersize=6,
                label="torch, round 2 (style B)")
        if rows_jb:
            xj = [r["n_copies"] for r in rows_jb]
            yj = [r.get(key) or r["env_steps_per_sec"] / r["n_copies"] for r in rows_jb]
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


def sweep_record():
    """The demonstration sweep's run record, if it has been produced."""
    hits = sorted(RUNS.glob("*sweep_demo*/data/*.json"))
    return json.loads(hits[-1].read_text()) if hits else None


def fig_sweep_curves():
    """Learning curves per learning-rate group from the demonstration sweep."""
    r = sweep_record()
    if not r:
        return pending("sweep figure", "the demonstration sweep record")
    import numpy as np
    g = np.array(r["group_index"])
    hist = r["history"]
    # a learning rate is an ordered quantity, so the groups take one sequential ramp
    # (light = smallest rate) rather than unrelated categorical hues
    ramp = ["#b7d3f6", "#86b6ef", "#3987e5", "#1c5cab", "#104281"]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), dpi=160)
    for gi, rate in enumerate(r["sweep_rates"]):
        idx = np.where(g == gi)[0]
        xs = [row["global_step"] / r["n_copies"] / 1e6 for row in hist]
        rew = [float(np.mean([row["reward_ext_sum_per_copy"][i] for i in idx])) for row in hist]
        cov = [float(np.mean([row["coverage_per_copy"][i] for i in idx])) for row in hist]
        color = ramp[min(gi, len(ramp) - 1)]
        axes[0].plot(xs, rew, "-", color=color, linewidth=1.7, label=f"rate {rate:g}")
        axes[1].plot(xs, cov, "-", color=color, linewidth=1.7, label=f"rate {rate:g}")
    axes[0].set_ylabel("extrinsic reward per copy per iteration (group mean)")
    axes[1].set_ylabel("maze coverage (fraction of open cells, group mean)")
    for ax in axes:
        ax.set_xlabel("environment steps per copy [millions]")
        ax.legend(frameon=False, fontsize=9)
        style_ax(ax)
    fig.suptitle(f"One run, {len(r['sweep_rates'])} learning rates, "
                 f"{r['copies_per_rate'][0]} independent copies each", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGS / "sweep_curves.png")
    plt.close(fig)
    return None


def fig_uniform_vs_sweep():
    """Cost of one rate for everything, against a per-copy rate vector, against differing rates."""
    d = newest(r"uniform_vs_sweep")
    if not d or not d["rows"]:
        return pending("uniform-versus-sweep figure", "the uniform_vs_sweep JSON")
    rows = d["rows"]
    xs = [r["total_copies"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3), dpi=160)
    # left: the three arms, absolute
    for label, key, color, ls, mk in [
            ("one rate for everything (torch fused Adam)", "uniform_sec", C_TORCH, "-", "o"),
            ("per-copy rate vector, all rates equal", "same_rates_sec", C_CUDA, "--", "s"),
            ("per-copy rate vector, rates differ", "diff_rates_sec", C_JAX, ":", "^")]:
        axes[0].plot(xs, [r[key] * 1e3 for r in rows], ls, color=color, linewidth=2,
                     marker=mk, markersize=6, label=label)
    axes[0].set_xscale("log", base=2)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("milliseconds per iteration")
    # right: the two costs as percentages, against the measured noise band
    axes[1].axhline(0.0, color="#9aa0ab", linewidth=1, linestyle=":")
    noise = [r["noise_floor_sec"] / r["uniform_sec"] * 100 for r in rows]
    axes[1].fill_between(xs, [-n for n in noise], noise, color="#d9d9d9", alpha=0.7,
                         label="noise floor (repeat spread)")
    axes[1].plot(xs, [r["vector_cost_percent"] for r in rows], "--", color=C_CUDA,
                 linewidth=2, marker="s", markersize=6,
                 label="cost of the per-copy vector (the optimizer change)")
    axes[1].plot(xs, [r["differing_cost_percent"] for r in rows], ":", color=C_JAX,
                 linewidth=2, marker="^", markersize=6,
                 label="cost of the rates actually differing")
    axes[1].set_xscale("log", base=2)
    axes[1].set_ylabel("percent slower than one rate for everything")
    for ax in axes:
        ax.set_xlabel("total copies (log scale)")
        ax.legend(frameon=False, fontsize=8)
        style_ax(ax)
    fig.suptitle("What a sweep costs, separated into its two parts", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGS / "uniform_vs_sweep.png")
    plt.close(fig)
    return None


def sec_uniform_vs_sweep():
    """The uniform-versus-swept comparison, decomposed."""
    d = newest(r"uniform_vs_sweep")
    if not d or not d["rows"]:
        return "### One rate for everything, against a sweep\n" + \
            pending("uniform-versus-sweep", "the uniform_vs_sweep JSON")
    rows = d["rows"]
    md = f"""### One rate for everything, against a sweep

Comparing a uniform run with a swept run mixes two changes, so they are measured apart. Giving
each copy its own learning rate makes the rate a VECTOR, and torch's fused Adam takes one
scalar rate per parameter group — so a sweep runs a hand-written batched Adam instead. That is
a real code change. The rates then actually DIFFERING changes no code at all: the same kernels
read different constants. The three arms below are matched in total copies, {d['n_rates']}
groups in the swept arms, measured in ABBA order in separate processes.

| total copies | one rate [ms] | rate vector, equal rates [ms] | rate vector, differing rates [ms] | noise floor [ms] | cost of the vector | cost of differing |
|---|---|---|---|---|---|---|
"""
    for r in rows:
        md += (f"| {r['total_copies']} | {r['uniform_sec']*1e3:.2f} | "
               f"{r['same_rates_sec']*1e3:.2f} | {r['diff_rates_sec']*1e3:.2f} | "
               f"{r['noise_floor_sec']*1e3:.2f} | {r['vector_cost_percent']:+.1f}% | "
               f"{r['differing_cost_percent']:+.1f}% |\n")
    diff = [abs(r["differing_cost_percent"]) for r in rows]
    vec = [r["vector_cost_percent"] for r in rows]
    md += f"""
![uniform versus sweep](figures/uniform_vs_sweep.png)

Reading the two columns:

- **The rates differing costs nothing**, as it must: at most {max(diff):.1f}% across the whole
  range, inside the noise floor at every point. Once the rate is a vector, whether its entries
  are equal or spread over three orders of magnitude changes only the numbers flowing through
  the same kernels.
- **The vector itself costs between {min(vec):+.1f}% and {max(vec):+.1f}%.** This is the
  optimizer change, and it is the only real price of being able to sweep. It is small because
  the per-copy Adam is compiled; before compiling it, it cost 38% (see the trainer ledger,
  round-3 rows).

So the practical answer is that a sweep is not a different regime from a uniform run — it is
the same run with a vector where a scalar used to be, and it is priced accordingly.
"""
    return md


def sweep_strategy_rows():
    """Every fused-versus-separate measurement, one per rate count, oldest first."""
    out = []
    for p in sorted(RESULTS.glob("*_sweep_strategies.json")):
        d = json.loads(p.read_text())
        fused = next((r for r in d["rows"] if r["strategy"] == "fused"), None)
        sep = next((r for r in d["rows"] if r["strategy"] == "separate"), None)
        uni = next((r for r in d["rows"] if r["strategy"] == "uniform"), None)
        if fused and sep and uni:
            out.append({"n_rates": fused["groups"], "copies_per_rate": fused["copies_per_rate"],
                        "total_copies": fused["total_copies"],
                        "fused_ms": fused["sec_per_iteration"] * 1e3,
                        "separate_ms": sep["sec_per_iteration"] * 1e3,
                        "uniform_ms": uni["sec_per_iteration"] * 1e3,
                        "speedup": sep["sec_per_iteration"] / fused["sec_per_iteration"],
                        "overhead_vs_uniform": fused["sec_per_iteration"] / uni["sec_per_iteration"] - 1})
    best = {}
    for r in out:                       # keep the newest measurement per rate count
        best[r["n_rates"]] = r
    return sorted(best.values(), key=lambda r: r["n_rates"])


def fig_sweep_vs_separate():
    """How much fusing the groups wins, as the number of rates grows."""
    rows = sweep_strategy_rows()
    if len(rows) < 2:
        return pending("fused-versus-separate figure", "more sweep strategy JSONs")
    fig, ax = plt.subplots(figsize=(6.6, 4.2), dpi=160)
    ax.plot([r["n_rates"] for r in rows], [r["speedup"] for r in rows], "-", color=C_TORCH,
            linewidth=2, marker="o", markersize=7, label="measured")
    ax.axhline(1.0, color="#9aa0ab", linewidth=1, linestyle=":")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("number of learning rates (128 copies each, log scale)")
    ax.set_ylabel("times faster than training the groups one after another")
    ax.set_ylim(bottom=0.9)
    ax.legend(frameon=False, fontsize=9)
    style_ax(ax)
    fig.suptitle("Fusing the rate groups into one run", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGS / "sweep_vs_separate.png")
    plt.close(fig)
    return None


def sweep_scaling(study):
    """Rows of one sweep-scaling study, newest run."""
    d = newest(rf"sweep_scaling_{study}")
    return d["rows"] if d else []


def fig_sweep_scaling():
    """Throughput against total copies for both sweep-scaling studies, and the wall-clock view."""
    rates, copies = sweep_scaling("rates"), sweep_scaling("copies")
    if not rates:
        return pending("sweep scaling figures", "the sweep scaling JSONs")
    uni = (newest(r"trainbench_torch_epoch_minibatch_round2b_styleB") or {}).get("rows", [])
    uni += (newest(r"trainbench_torch_epoch_minibatch_round2c_large") or {}).get("rows", [])
    uni = sorted({r["n_copies"]: r for r in uni}.values(), key=lambda r: r["n_copies"])

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3), dpi=160)
    series = [("more learning rates (128 copies each)", C_TORCH, "-", "o", rates),
              ("more copies per rate (16 rates)", C_CUDA, "--", "s", copies)]
    # total throughput
    for label, color, ls, mk, rows in series:
        axes[0].plot([r["total_copies"] for r in rows], [r["env_steps_per_sec"] for r in rows],
                     ls, color=color, linewidth=2, marker=mk, markersize=6, label=label)
    if uni:
        axes[0].plot([r["n_copies"] for r in uni],
                     [r["env_steps_per_sec"] for r in uni], ":", color=C_JAX, linewidth=1.8,
                     marker="^", markersize=5, label="one learning rate (no sweep)")
    axes[0].set_ylabel("total environment steps / second")
    # per-copy throughput, with the per-environment scale on the right
    for label, color, ls, mk, rows in series:
        axes[1].plot([r["total_copies"] for r in rows],
                     [r["env_steps_per_sec_per_copy"] for r in rows],
                     ls, color=color, linewidth=2, marker=mk, markersize=6, label=label)
    if uni:
        axes[1].plot([r["n_copies"] for r in uni],
                     [r["env_steps_per_sec_per_copy"] for r in uni], ":", color=C_JAX,
                     linewidth=1.8, marker="^", markersize=5, label="one learning rate (no sweep)")
    axes[1].set_ylabel("environment steps / second per copy")
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("total copies (log scale)")
        ax.legend(frameon=False, fontsize=8.5)
        style_ax(ax)
    # the same measurement in per-environment units: a linked secondary axis stays aligned
    # with the left one whatever the autoscaling does (one copy holds n_envs environments)
    right = axes[1].secondary_yaxis("right", functions=(lambda y: y / 4.0, lambda y: y * 4.0))
    right.set_ylabel("environment steps / second per environment", color="#5c6270")
    right.tick_params(colors="#5c6270")
    fig.suptitle("Sweep scaling: adding rates costs exactly what adding copies costs",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGS / "sweep_scaling.png")
    plt.close(fig)

    # wall clock to a realistic budget, which is what decides whether a sweep is affordable
    fig, ax = plt.subplots(figsize=(6.6, 4.2), dpi=160)
    iters = 10e6 / (128 * 4)
    for label, color, ls, mk, rows in series:
        ax.plot([r["total_copies"] for r in rows],
                [r["sec_per_iteration"] * iters / 3600 for r in rows],
                ls, color=color, linewidth=2, marker=mk, markersize=6, label=label)
    sep = [(r["n_rates"], rates[0]["sec_per_iteration"] * r["n_rates"] * iters / 3600)
           for r in rates]
    ax.plot([r["total_copies"] for r in rates], [y for _, y in sep], "-.", color="#e34948",
            linewidth=1.8, marker="x", markersize=6,
            label="the same groups trained one after another")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("total copies (log scale)")
    ax.set_ylabel("hours to 10 million environment steps per copy")
    ax.legend(frameon=False, fontsize=8.5)
    style_ax(ax)
    fig.suptitle("What a sweep actually costs in wall time", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGS / "sweep_wallclock.png")
    plt.close(fig)
    return None


def sec_sweep_scaling():
    """How the sweep scales in rates and in copies."""
    rates, copies = sweep_scaling("rates"), sweep_scaling("copies")
    groups = sweep_scaling("groups")
    if not rates:
        return "### How it scales\n" + pending("sweep scaling", "the sweep scaling JSONs")
    iters = 10e6 / (128 * 4)
    head = next((r for r in rates if r["n_rates"] == 16 and r["copies_per_rate"] == 128), None)
    md = ""
    if head:
        sep = next((r for r in sweep_strategy_rows() if r["n_rates"] == 16), None)
        hours = head["sec_per_iteration"] * iters / 3600
        md += f"""### The configuration you asked about

Sixteen learning rates with 128 independent copies each — 2,048 copies in one run — takes
**{head['sec_per_iteration']*1e3:.0f} ms per iteration**: {sci(head['env_steps_per_sec'])}
environment steps per second in total, {head['env_steps_per_sec_per_copy']:.0f} per copy,
{head['env_steps_per_sec_per_env']:.0f} per environment, in {head['peak_vram_mb']/1024:.1f} GB.
Training all 2,048 of them for 10 million environment steps each takes **{hours:.2f} hours**"""
        if sep:
            md += (f", against {sep['separate_ms']/head['sec_per_iteration']/1e3*hours:.2f} hours "
                   f"if the sixteen groups were trained one after another ({sep['speedup']:.2f}x)")
        md += ".\n\n"
    md += """### How it scales

Two knobs grow a sweep: the number of learning rates, and the copies behind each rate. Both
were measured, and they give the same curve — because both only change the total copy count,
which is the only thing the cost depends on.

| rates | copies per rate | total copies | ms/iteration | total env-steps/s | per copy | per env | peak VRAM [MB] | hours to 10M steps/copy |
|---|---|---|---|---|---|---|---|---|
"""
    for r in rates + copies:
        md += (f"| {r['n_rates']} | {r['copies_per_rate']} | {r['total_copies']} | "
               f"{r['sec_per_iteration']*1e3:.1f} | {sci(r['env_steps_per_sec'])} | "
               f"{r['env_steps_per_sec_per_copy']:.0f} | {r['env_steps_per_sec_per_env']:.0f} | "
               f"{r['peak_vram_mb']:.0f} | {r['sec_per_iteration']*iters/3600:.2f} |\n")
    md += """
The first six rows grow the rate count at 128 copies each; the last six grow the copies per
rate at 16 rates. Read them against each other: at every total copy count the two agree to
0.3 ms out of as much as 274 ms, which is also the best noise estimate available here — the
two studies are the same measurement reached from different directions.

![sweep scaling](figures/sweep_scaling.png)

Per-environment throughput is the per-copy column divided by the four environments each copy
holds; it is drawn as the right-hand scale of the second panel rather than as its own curve,
because it is the same measurement in different units.

"""
    if groups:
        ms = [r["sec_per_iteration"] * 1e3 for r in groups]
        spread = max(ms) - min(ms)
        md += f"""### Does the number of rates cost anything by itself?

No. Holding the total at {groups[0]['total_copies']} copies and splitting them into more and
more groups leaves the time per iteration flat and the memory byte-identical:

| groups (learning rates) | copies per rate | copies | ms/iteration | million steps/s | thousand steps/s per copy | hours per million steps per copy | peak VRAM [MB] |
|---|---|---|---|---|---|---|---|
"""
        for r in groups:
            md += (f"| {r['n_rates']} | {r['copies_per_rate']} | {r['total_copies']:,} | "
                   f"{r['sec_per_iteration']*1e3:.1f} | {r['env_steps_per_sec']/1e6:.2f} | "
                   f"{r['env_steps_per_sec_per_copy']/1e3:.2f} | "
                   f"{H(r['env_steps_per_sec_per_copy'])} | "
                   f"{r['peak_vram_mb']:.0f} |\n")
        md += f"""
From 1 group to {groups[-1]['n_rates']} groups the spread is {spread:.2f} ms on a mean of
{sum(ms)/len(ms):.1f} ms ({spread/(sum(ms)/len(ms))*100:.2f}%), which is within the run-to-run
noise — and the study was run twice, once from 1 group upward and once from 128 groups
downward, so the flatness is not an artifact of measuring the points in a fixed order. This is
what the implementation predicts: the learning rate is a vector indexed by copy,
Adam is elementwise, the gradient clip reduces per copy, and the loss sums over copies — so
nothing in the computation is aware of how many distinct rate VALUES that vector holds. Sixteen
rates cost what sixteen times as many copies cost, and nothing more.

![sweep wall clock](figures/sweep_wallclock.png)

"""
    strat = sweep_strategy_rows()
    if strat:
        md += """### Fusing the groups, against running them one after another

| rates | copies per rate | fused [ms] | separate [ms] | uniform rate, same size [ms] | fused is faster by | sweep overhead vs uniform |
|---|---|---|---|---|---|---|
"""
        for r in strat:
            md += (f"| {r['n_rates']} | {r['copies_per_rate']} | {r['fused_ms']:.1f} | "
                   f"{r['separate_ms']:.1f} | {r['uniform_ms']:.1f} | {r['speedup']:.2f}x | "
                   f"{r['overhead_vs_uniform']*100:+.1f}% |\n")
        md += """
The advantage of fusing grows with the number of rates and flattens out: each group on its own
is too small to fill the GPU, so running them in turn wastes most of the machine, while fusing
them turns the whole sweep into one batched computation. The last column is the one that
matters for planning: a sweep costs the same as training the same number of copies at a single
rate, to within about one percent either way.

![fused versus separate](figures/sweep_vs_separate.png)
"""
    return md


def sec_sweep():
    """Round 3: sweeping a hyperparameter across copy groups."""
    md = """## Sweeping learning rates across copy groups

The trainer's copies were identical apart from their seed. They can now be split into groups,
each group training at its own learning rate, in one batched run — the inputs are exactly the
rates to try and how many copies each rate gets:

```python
cfg = sweep_config(learning_rates=[1e-4, 3e-4, 1e-3, 3e-3], copies_per_rate=128)
```

Adam is elementwise, so copies stacked on the leading axis are already independent optimizers;
the only thing `torch.optim.Adam` cannot express is a different rate per copy, so a sweep
switches to a hand-written batched Adam with torch's formula and a per-copy rate vector. Copy k
of every group starts from the same weights and meets the same environments (paired seeding),
so a difference between groups is the rate's doing. Full description: `ppo/torch_ppo/SWEEP.md`.

### Which layout is best

Three ways to arrange G groups of K copies, measured at 4 rates x 128 copies = 512 copies:

| layout | copies | ms per sweep iteration | million steps/s | thousand steps/s per copy | hours per million steps per copy | peak VRAM [MB] |
|---|---|---|---|---|---|---|
"""
    d = newest(r"sweep_strategies")
    if not d:
        return md + pending("sweep layout table", "the sweep strategy JSON")
    names = {"uniform": "uniform rate, one trainer (not a sweep — the reference)",
             "fused": "one trainer, per-copy rate vector, one graph",
             "separate": "one trainer per rate, run in turn"}
    # this file records only the iteration time, so the rates are derived from it: one iteration
    # advances every copy by num_steps x n_envs = 512 environment steps, and the hours column is
    # that per-copy rate written as time
    # before: {"total_copies": 2048, "sec_per_iteration": 0.1445}
    # after:  7.26 million steps/s in total, 3.54 thousand steps/s per copy, 0.078 hours per
    #         million steps per copy
    for row in d["rows"]:
        per_copy = STEPS_PER_COPY_PER_ITER / row["sec_per_iteration"]
        md += (f"| {names[row['strategy']]} | {row['total_copies']:,} | "
               f"{row['sec_per_iteration']*1e3:.2f} | "
               f"{per_copy*row['total_copies']/1e6:.2f} | {per_copy/1e3:.2f} | "
               f"{H(per_copy)} | {row['peak_vram_mb']:.0f} |\n")
    md += f"""
Fusing the groups into one batched run is {d['fused_speedup_vs_separate']:.2f}x faster than
running them one after another, and costs {d['fused_overhead_vs_uniform']*100:+.1f}% against a
uniform-rate run of the same total size — so a sweep is very nearly free relative to training
the same number of copies at a single rate. Running the groups separately is slower for the
reason the whole project rests on: each group alone is too small to fill the GPU, and the
per-iteration cost is dominated by kernel count rather than by arithmetic.

### It finds the right answer

A demonstration run of 5 rates x 32 copies (3.07M environment steps per copy, 140 seconds):

"""
    r = sweep_record()
    if r:
        import numpy as np
        g = np.array(r["group_index"])
        hist = r["history"]
        late = hist[-len(hist) // 5:]
        md += ("| learning rate | copies | reward per iteration, last 20% | final coverage | "
               "copies that reached the goal |\n|---|---|---|---|---|\n")
        for gi, rate in enumerate(r["sweep_rates"]):
            idx = np.where(g == gi)[0]
            rew = float(np.mean([[row["reward_ext_sum_per_copy"][i] for i in idx]
                                 for row in late]))
            cov = float(np.mean([r["final_coverage_per_copy"][i] for i in idx]))
            ever = sum(1 for i in idx
                       if max(row["reward_ext_sum_per_copy"][i] for row in hist) > 0)
            md += f"| {rate:g} | {len(idx)} | {rew:.2f} | {cov:.3f} | {ever}/{len(idx)} |\n"
        md += """
The sweep recovers the expected shape — too small a rate barely learns, too large a rate is
unstable, and the best value is the 3e-4 the project had been using — from a single run that
cost the same as training those copies at one rate.

![sweep curves](figures/sweep_curves.png)
"""
    return md


def campaign_records(round2=True):
    """The final-campaign per-count JSONs, {(style, C): record}, for one round's campaign."""
    pattern = "2026-*final_round2*/data/copies_*.json" if round2 else \
              "2026-08-15-02-56_final*/data/copies_*.json"
    out = {}
    for p in RUNS.glob(pattern):
        r = json.loads(p.read_text())
        out[(r["style"], r["n_copies"])] = r
    return out


def fig_campaign_curves():
    """Learning curves of the final campaign: per-copy-count mean reward and coverage."""
    recs = campaign_records()
    styleb = {c: r for (s, c), r in recs.items() if s == "epoch_minibatch"}
    if not styleb:
        return pending("campaign figures", "final-run JSONs")
    # one blue-ramp shade per copy count (sequential: darker = more copies)
    ramp = ["#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#104281"]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), dpi=160)
    for i, c in enumerate(sorted(styleb)):
        h = styleb[c]["history"]
        xs = [row["global_step"] / c / 1e6 for row in h]
        rew = [sum(row["reward_ext_sum_per_copy"]) / c for row in h]
        cov = [sum(row["coverage_per_copy"]) / c for row in h]
        color = ramp[min(i, len(ramp) - 1)]
        axes[0].plot(xs, rew, "-", color=color, linewidth=1.6, label=f"C={c}")
        axes[1].plot(xs, cov, "-", color=color, linewidth=1.6, label=f"C={c}")
    axes[0].set_ylabel("extrinsic reward per copy per iteration (mean over copies)")
    axes[1].set_ylabel("maze coverage (fraction of open cells, mean over copies)")
    for ax in axes:
        ax.set_xlabel("environment steps per copy [millions]")
        ax.legend(frameon=False, fontsize=9)
        style_ax(ax)
    fig.suptitle("Final campaign (style B): learning vs copy count", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGS / "campaign_curves.png")
    plt.close(fig)
    return None


# ---------------- report sections ----------------

def sec_overview():
    return """# GPU-parallel PointMaze + batched multi-copy PPO+RND — unified report

All numbers were measured on serval05 (one NVIDIA H100 NVL, 95 GB; direct ssh, exclusive use
enforced by a lock), from the code in this repository under `09_parallelization/`. Every table
cell traces to a JSON in `benchmarks/results/` or a run record under `train_runs/`; this report
is generated by `report/.../code/make_report.py` and can be regenerated from those files.

The work built three modules and then improved them over three passes of the same loop —
baseline, one change, measure, keep or revert, write a row:

| pass | what it did | headline |
|---|---|---|
| Round 1 | build everything: exact GPU environments in PyTorch, fused CUDA and JAX; the batched multi-copy PPO+RND trainer in PyTorch and JAX; end-to-end capture | 777 -> 36.6 ms per training iteration at 128 copies |
| Round 2 | improve on the finished system, with a measurement protocol that can tell a small change from drift | 36.8 -> 20.2 ms, a further 1.8x |
| Round 3 | new capability: train copy groups at DIFFERENT learning rates in one run | a sweep costs 1.0% more than a uniform run of the same size |
| Round 4 | close the distance to the JAX trainer at 8 to 128 copies | 20.4 -> 16.3 ms at 128 copies, and a loss at 512 that round 5 traced |
| Round 5 | re-open the question at the copy counts the trainer is used at, 1,024 to 4,096 | the limit changes from the number of device programs to memory traffic, and the previous round's loss turns out to be a regression of up to 19 percent there |

Reading order: what was built and why it is correct, then the module-by-module numbers, then
what each round changed, then the training campaign and the sweep. The last section is the
newest and stands somewhat apart: it is about the sizes the trainer is actually run at, where
the earlier sections' conclusions do not all carry over.
"""


def sec_correctness():
    return """## Environment correctness (all three implementations)

The environments are not approximations: MuJoCo's exact per-step update was extracted and
verified, including the contact model (solref/solimp force law, margin semantics, and the
joint two-contact solve, all probe-verified against the solver's internal arrays).

1. One-step (teacher-forced) error vs the reference MuJoCo env is at machine precision
   (float64 <= 4.4e-16 on every fixture case, including wall slides across box boundaries
   and corner wraps; float32 <= 5e-7).
2. Closed-loop 400-step multi-contact rollouts match the reference to 1e-13 (one fixture
   passes through a knife-edge equilibrium — ball balanced on a corner — where closed-loop
   paths legitimately separate on rounding noise; its one-step check stays exact).
3. Reset randomness is a frozen counter-based generator: torch, CUDA, and JAX produce
   BIT-IDENTICAL resets (verified including post-truncation respawns).
4. The CUDA kernel's trajectories match the torch env within 2.1e-7 over 500 steps with
   320 auto-resets (mostly bit-equal; rare 1-2 ULP differences on contact steps).
5. A 256-episode random-policy comparison against the reference env passes (visit
   distribution total-variation distance 0.026, gate 0.05).
"""


def sec_module1():
    md = "## Module 1 — environment throughput\n\n"
    tab = []
    for label, pat in [("torch eager", r"envbench_torch_eager\."),
                       ("torch compiled", r"envbench_torch_compile_grid"),
                       ("CUDA fused kernel", r"envbench_cuda_fused_tourn_count"),
                       ("jax jit per step", r"envbench_jax_jit_grid"),
                       ("jax scan (fused rollout, upper bound)", r"envbench_jax_scan_grid")]:
        d = newest(pat)
        if not d:
            continue
        rows = {r["total_envs"]: r for r in d["rows"]}
        cells = [label]
        for n in (1000, 100000, 1000000, 3000000):
            r = rows.get(n)
            cells.append(sci(r["env_steps_per_sec"]) if r else "—")
        peak = max(d["rows"], key=lambda r: r["env_steps_per_sec"])
        cells.append(f"{sci(peak['env_steps_per_sec'])} at {peak['total_envs']:,}")
        tab.append(cells)
    if not tab:
        return md + pending("Module 1 table", "env grid JSONs")
    md += ("| implementation | 1e3 envs | 1e5 envs | 1e6 envs | 3e6 envs | peak |\n"
           "|---|---|---|---|---|---|\n")
    best = {i: max(row[i] for row in tab if row[i] != "—") for i in range(1, 5)}
    for row in tab:
        md += "| " + " | ".join(row) + " |\n"
    md += """
Environment steps per second, single batch axis (n_copies folded in). Figures:

![env throughput](figures/env_throughput.png)

![env step time](figures/env_steptime.png)

Where the per-step time starts to RISE (the log-scale line's knee — before it, adding
environments is free because the step is launch-overhead-bound; after it, kernel time
dominates and the time grows with the batch):

- torch compiled: flat ~155-171 us to 3e5 envs; rises from ~3e5.
- CUDA fused kernel: flat ~6 us to 3e4 envs; rises from ~1e5 (it reaches the
  memory-bandwidth regime earliest because one kernel has no launch amortization to hide).
- jax jit: flat ~71-79 us to ~1e5; rises past 1e5.
- jax scan: flat ~15-19 us to ~1e5; peak total throughput lands at 3e5 envs (L2-cache
  residency) and settles slightly lower at HBM scale.

Per-env step speed at 1e3 envs (batched step time divided by env count): CUDA 6.0 ns per
env-step, jax scan 14.8 ns, jax jit 79 ns, torch compiled 158 ns. All fall toward the memory
bandwidth limit as the batch grows: at 3e6 envs the CUDA kernel spends 57 picoseconds per
env-step (171.7 us for the whole batch).
"""
    return md


def sec_module2():
    md = """## Module 2 — batched multi-copy trainer throughput

Both trainers implement the same algorithm (spec: `ppo/research/ppo_rnd_algorithm_spec.md`),
verified by bitwise copy-isolation tests in each framework and a cross-framework forward
agreement of 8.6e-7. Style A = one full-batch update per rollout; style B = 4 epochs x 4
minibatches. T=128, N=4 (512 rows per copy per iteration).

| trainer | style | C=8 | C=128 | notes |
|---|---|---|---|---|
"""
    packb = newest(r"trainbench_torch_epoch_minibatch_pack\.")
    packa = newest(r"trainbench_torch_full_batch_pack")
    base = newest(r"trainbench_torch_epoch_minibatch\.json")
    jax0 = newest(r"trainbench_jax_ppo\.")
    jaxd = newest(r"trainbench_jax_ppo_donate")

    def row(d, style, note, cfilter=(8, 128)):
        if not d:
            return ""
        r = {x["n_copies"]: x for x in d["rows"] if x.get("style", style) == style}
        cells = []
        for c in cfilter:
            x = r.get(c)
            spi = x and (x.get("sec_per_iteration") or x.get("sec_per_iteration_median"))
            cells.append(f"{spi*1e3:.1f} ms = {sci(x['env_steps_per_sec'])}/s" if x else "—")
        return f"| {d['impl']} | {style} | {cells[0]} | {cells[1]} | {note} |\n"

    md += row(base, "epoch_minibatch", "BEFORE any optimization (eager)")
    md += row(packb, "epoch_minibatch", "AFTER (one-graph capture + TF32 + hoist + packing)")
    if packa:
        x = packa["rows"][0]
        md += (f"| torch_ppo | full_batch | — | {x['sec_per_iteration']*1e3:.1f} ms = "
               f"{sci(x['env_steps_per_sec'])}/s | AFTER (same config) |\n")
    if jaxd:
        for st in ("full_batch", "epoch_minibatch"):
            md += row(jaxd, st, "jax whole-iteration jit + donation")
    md += """
The eager torch baseline is 777 ms per iteration at every copy count; the final torch
configuration is 21x faster at C=128. The jax trainer reaches 74 iter/s (style A) and 50
iter/s (style B) at C=128. A LABELED algorithm variant (T=32, N=16 — same 512 rows, shorter
GAE horizon) reaches 226 iter/s = 1.48e7 env-steps/s in jax style A; it is recorded as a
variant, not the default, because it changes the algorithm.
"""
    return md


def sec_module3():
    md = """## Module 3 — end-to-end fusion and the pairing grid

- torch env + torch trainer: the production path. The environment's pure `step_core`
  composes into the compiled per-step function; one CUDA graph captures the WHOLE iteration
  (128-step rollout + statistics/GAE post-processing + the 16-step update incl. Adam), so
  one replay per iteration and zero python in the loop. Captured paths are verified
  BITWISE equal to the uncaptured trainer.
- jax env + jax trainer: the whole iteration is one jitted XLA program with donated state.
- CUDA env + torch trainer (`env_backend="cuda"`): the fused kernel replaces the env part
  of the captured rollout, with the policy and RND parts as compiled subgraphs around it.
"""
    cud = newest(r"trainbench_torch_epoch_minibatch_cudaenv_streamfixed_prod")
    tor = newest(r"trainbench_torch_epoch_minibatch_round2b_styleB")
    if cud and cud["rows"] and tor:
        cc = {r["n_copies"]: r["sec_per_iteration"] * 1e3 for r in cud["rows"]}
        tc = {r["n_copies"]: r["sec_per_iteration"] * 1e3 for r in tor["rows"]}
        parts = [f"C={c}: {cc[c]:.1f} ms with the CUDA kernel versus {tc[c]:.1f} ms with the "
                 f"PyTorch environment" for c in sorted(cc) if c in tc]
        md += "  Measured in the shipped configuration (style B): " + "; ".join(parts) + ".\n"
        md += """  The two are the same to about 1.5%, which is the point worth taking away: after
  round two moved most per-step work out of the rollout loop, the environment is a small part
  of a training iteration, so an environment kernel that is 24x faster on its own changes the
  training rate very little. The fast kernel earns its place in environment-only work (data
  generation, evaluation sweeps), not in this trainer.\n"""
    else:
        md += pending("Module 3 cuda pairing", "the re-measured pairing on the stream-fixed kernel")
    md += """  An earlier pairing measurement was discarded: the environment kernels launched on
  the legacy default stream rather than the capture stream, so the captured graph contained no
  environment step at all (PyTorch reported an empty graph once a test looked for it). The
  kernel now launches on the current stream and a test replays a captured env step and checks
  the state actually advanced.
"""
    md += """- Cross-framework pairings (jax env + torch trainer or the reverse) were MEASURED over
  the dlpack boundary: crossing frameworks costs +6.2 to +7.0 ms per environment step on
  top of a 70-260 us native step (about 40-90x), because every crossing forces a
  synchronization between the two runtimes — in addition to structurally breaking
  CUDA-graph capture and XLA scan fusion. They are documented as non-production paths
  (`benchmarks/bench_cross_pairing.py`).
"""
    return md


def sec_copies():
    md = """## Scaling the number of independent training copies

The task's two questions, answered by measurement (style B, T=128, N=4, final config):

1. **Does increasing the copy count decrease per-copy throughput?** Yes, but only past a
   knee: per-copy throughput is nearly flat up to C~128-256, then decays roughly as 1/C.
2. **Does increasing the copy count decrease TOTAL throughput?** No. Total throughput
   rises monotonically and saturates; it never goes down up to the memory limit.

| copies C | round 1 ms/iter | round 2 ms/iter | round 2 total env-steps/s | round 2 per-copy env-steps/s |
|---|---|---|---|---|
"""
    r1 = {r["n_copies"]: r for r in torch_cscale_rows(round2=False)}
    r2 = {r["n_copies"]: r for r in torch_cscale_rows(round2=True)}
    for c in sorted(set(r1) | set(r2)):
        a = f"{r1[c]['sec_per_iteration']*1e3:.1f}" if c in r1 else "—"
        if c in r2:
            b = f"{r2[c]['sec_per_iteration']*1e3:.1f}"
            tot = sci(r2[c]["env_steps_per_sec"])
            per = sci(r2[c]["env_steps_per_sec_per_copy"])
        else:
            b = tot = per = "—"
        md += f"| {c} | {a} | {b} | {tot} | {per} |\n"
    md += """
**How many copies fit, and the trade the second round made.** Round one reached 32,768 copies
and ran out of memory at 65,536; round two runs out at 32,768. That is not a regression to fix
by accident — it is the price of the speed: hoisting the per-step work into wide passes, and
caching the frozen RND target's features for the update phase, both hold more data at once
(the cached features alone are 256 KB per copy). The ceiling moved from 32,768 copies to
16,384 while every copy count up to there got about 1.8x faster. A run that needs the extra
copies more than the speed can set `compile_post=False` and skip the hoist to get the round-one
memory profile back. The ceiling is genuine demand rather than fragmentation: retrying 32,768
copies with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` reduced the failing allocation
from 32 GiB to 8 GiB but still ran out.

Round-two total throughput saturates near 7.8e6 environment steps per second from about 8,192
copies (round one: 5.6e6). The jax trainer saturates near 1.9e7 (style A) / 9.1e6 (style B)
around 2,048-4,096 copies.

![copy scaling](figures/copy_scaling.png)
"""
    return md


def sec_profile():
    d = newest(r"profile_breakdown")
    if not d:
        return "## Profiling breakdown\n" + pending("profiling", "profile JSON")
    md = """## Profiling breakdown (C=128)

Production view — what one training iteration actually pays with the capture
configuration measured (separate rollout/update graphs, eager post-processing; the
one-graph mode removes most of the post-processing row's launch gaps):

| phase | time [ms] | share |
|---|---|---|
"""
    total = sum(d["production_us"].values())
    for k, v in d["production_us"].items():
        md += f"| {k} | {v/1000:.2f} | {v/total*100:.1f}% |\n"
    md += f"| TOTAL | {total/1000:.2f} | 100% |\n"
    md += """
Component view — each logical block called in isolation with EAGER kernels (per-iteration
equivalents). This view shows where the raw work lives before fusion; it deliberately
over-weights blocks whose eager form launches many small kernels (the compiled/captured
production path fuses most of them away):

| block | eager time [ms] | share |
|---|---|---|
"""
    # dynamics_step is a SUBSET of step_core; keep it out of the denominator and show it
    # as an of-which row so the shares sum to 100%
    comp = dict(d["component_us"])
    dyn_key = next(k for k in comp if "dynamics_step" in k)
    dyn = comp.pop(dyn_key)
    ctot = sum(comp.values())
    for k, v in sorted(comp.items(), key=lambda kv: -kv[1]):
        md += f"| {k} | {v/1000:.2f} | {v/ctot*100:.1f}% |\n"
        if "step_core" in k:
            md += (f"| — of which {dyn_key} | {dyn/1000:.2f} | "
                   f"({dyn/v*100:.0f}% of the env step) |\n")
    md += """
Reading: in eager form the ENVIRONMENT dominates (about 84% of raw kernel time), which is
why compiling/fusing the env step was the first large win; after fusion the three captured
phases are nearly balanced, and the remaining bottleneck at small C is fixed kernel time
in the 128 sequential rollout steps.
"""
    return md


def sec_before_after():
    """The task's before/after table: env, training(update), and end-to-end throughput for
    the final-run copy counts, before optimization (eager) vs the final configuration."""
    md = """## Before/after optimization at the final-run sizes (8-128 copies)

Environment-only throughput at the exact env-batch sizes the final runs use
(batch = C copies x 4 envs), env steps per second. "Eager" is the CURRENT physics code run
without compilation (the uncompiled per-step python cost, ~4.4 ms per batched step at
these sizes, is what the compiled/captured paths remove):

| env batch | eager (before) | compiled (after) | CUDA kernel (after) |
|---|---|---|---|
"""
    eag = newest(r"envbench_torch_eager_smallN")
    cmp_ = newest(r"envbench_torch_compile_smallN")
    cud = newest(r"envbench_cuda.*_smallN")
    if not (eag and cmp_):
        return md + pending("before/after env table", "smallN env JSONs")
    er = {r["total_envs"]: r for r in eag["rows"]}
    cr = {r["total_envs"]: r for r in cmp_["rows"]}
    ur = {r["total_envs"]: r for r in (cud or {"rows": []})["rows"]}
    for n in (32, 64, 128, 256, 512):
        cu = sci(ur[n]["env_steps_per_sec"]) if n in ur else "—"
        md += (f"| {n} | {sci(er[n]['env_steps_per_sec'])} | "
               f"{sci(cr[n]['env_steps_per_sec'])} | {cu} |\n")

    md += """
Whole-loop training throughput (environment steps per second while TRAINING, i.e. the
end-to-end number; before = eager baseline, after = the final one-graph configuration as
measured in the campaign itself):

| copies C | before, style B | after, style B | after, style A |
|---|---|---|---|
"""
    recs = campaign_records()
    base = newest(r"trainbench_torch_epoch_minibatch\.json")
    baser = {r["n_copies"]: r for r in (base or {"rows": []})["rows"]}
    for c in (8, 16, 32, 64, 128):
        b = baser.get(c)
        before = sci(b["env_steps_per_sec"]) if b else "7.77e2 ms/iter basis: " + sci(512 * c / 0.777)
        rb = recs.get(("epoch_minibatch", c))
        ra = recs.get(("full_batch", c))
        md += (f"| {c} | {sci(512 * c / 0.777)} | "
               f"{sci(rb['env_steps_per_sec']) if rb else '—'} | "
               f"{sci(ra['env_steps_per_sec']) if ra else '—'} |\n")

    md += """
(The before column uses the measured eager iteration time of 777 ms, which is flat across
copy counts.) Phase split of one iteration, before vs after (C=128, style B; the "after"
split is measured in the separate-graphs configuration — the shipped one-graph mode fuses
the phases and is faster than this sum):

| phase | before (eager) | after round 1 (captured) |
|---|---|---|
| rollout (env + policy interaction) | 680 ms | 20.1 ms |
| post-processing (GAE, statistics) | inside rollout | 14.8 ms (eager between graphs); inside the one-graph replay in the final config |
| update (16 minibatch steps) | 95 ms | 15.5 ms |
| whole iteration | 777 ms | 36.6 ms (one-graph) |

Round two then reworked what the loop does rather than how it is launched, and took the same
iteration from 36.8 ms to 20.2 ms at 128 copies (paired measurements, noise floor 0.01-0.10 ms):

| change | C=8 | C=128 |
|---|---|---|
| round-1 configuration (the starting point) | 26.6 ms | 36.8 ms |
| + compiled post-rollout body | 22.8 ms | 32.7 ms |
| + critic, log-probability and RND bonus hoisted out of the rollout loop | 14.7 ms | 21.2 ms |
| + per-step buffer writes folded into the compiled step | 14.0 ms | 20.2 ms |

Overall, one training iteration of 128 independent copies went from 777 ms to 20.2 ms — 38x —
while every change was verified to compute the same thing.
"""
    return md


def sec_campaign():
    recs = campaign_records()
    if not recs:
        return "## Final training campaign\n" + pending("campaign", "final-run JSONs")
    md = """## Final training campaign — 8 to 128 independent RND runs (PyTorch)

20,000 iterations per configuration = 10.24M environment steps per copy; PointMaze Large,
sparse goal, run-6 setup. Every copy is an independent seed with its own networks,
environments, statistics, and optimizer. The campaign was run twice — once on the round-1 code
and again on the round-2 code — so the improvement is visible on the deliverable itself and not
only on the benchmarks. The table reports the round-2 run, with the round-1 throughput beside
it for comparison.

| style | copies | wall time [s] | total env-steps/s | round-1 rate | env-steps/s per copy | copies at goal, late* | copies at goal, ever* | coverage mean | peak VRAM [MB] |
|---|---|---|---|---|---|---|---|---|---|
"""
    r1recs = campaign_records(round2=False)
    for (style, c) in sorted(recs, key=lambda k: (k[0], k[1])):
        r = recs[(style, c)]
        hist = r["history"]
        late = hist[-max(1, len(hist) // 10):]
        count = lambda rows: sum(1 for i in range(c)
                                 if max(row["reward_ext_sum_per_copy"][i] for row in rows) > 0)
        solved_late = count(late) if late else 0
        solved_ever = count(hist) if hist else 0
        cov = sum(r["final_coverage_per_copy"]) / c
        r1 = r1recs.get((style, c))
        md += (f"| {style} | {c} | {r['train_seconds']:.0f} | "
               f"{sci(r['env_steps_per_sec'])} | "
               f"{sci(r1['env_steps_per_sec']) if r1 else '—'} | "
               f"{sci(r['env_steps_per_sec'] / c)} | "
               f"{solved_late}/{c} | {solved_ever}/{c} | {cov:.3f} | "
               f"{r['peak_vram_mb']:.0f} |\n")
    md += """
The round-2 campaign also shows more copies reaching the goal and slightly higher coverage than
the round-1 one. That is NOT an effect of the optimization: the algorithm is unchanged and every
round-2 change was verified to compute the same function to within float32 rounding. What those
tiny differences do is send the runs down different trajectories, and on a sparse-goal task the
per-copy outcome is close to all-or-nothing, so the aggregate moves by more than one might
expect. Treat the two campaigns as two samples of the same procedure, and read the throughput
columns — not the goal columns — as the round's result.

*Both goal columns count copies that scored a positive extrinsic reward in a SAMPLED
iteration: "late" over the final 10% of samples, "ever" over all of them. Progress is sampled
every 50 iterations (400 samples of 20,000), so a copy that reached the goal only in an
unsampled iteration is missed and both columns are lower bounds. The gap between them is the
sparse goal being found and lost again rather than held (per-copy curves below).

![campaign curves](figures/campaign_curves.png)
"""
    return md


def sec_rounds():
    """What each round changed, and how it was measured."""
    return """## The first three rounds

### Round 1 — build it, and make it exact

The environments are not approximations of the reference: MuJoCo's per-step update and its
whole contact pipeline were extracted and verified to machine precision (correctness section
above). The trainer was written from a spec derived from CleanRL's PPO+RND, with every running
statistic, gradient clip and random draw made per copy so that copies cannot influence each
other. The speed came from compiling a fused per-step function and then capturing whole phases
as CUDA graphs, ending with the entire iteration — rollout, post-processing and update — as one
replay per iteration.

Round one compared configurations by running each once. That was good enough for changes worth
3x and useless for changes worth 3%.

### Round 2 — improve it, with a protocol that can tell a real change from drift

Every round-two change is measured by running both variants in ABBA order in separate
processes, comparing against the change's own predecessor git revision, and reporting the
within-variant spread as a noise floor; a difference below that floor is recorded as no effect.
Four research agents produced ranked experiment lists, an external-source sweep, an audit of
the performance checklist, and an adversarial review of the code and of the measurement
protocol.

What carried the gain:

- Compiling the post-rollout body, whose intrinsic-reward filter and two GAE scans were python
  loops over the horizon and therefore hundreds of tiny kernels: +11% at 128 copies, +14% at 8.
- Hoisting the critic forward, the log-probability and the RND bonus out of the 128-step
  rollout loop into single wide passes afterwards: +42% and +45%. Each is a pure function of
  data the loop already stores and of parameters that do not change during a rollout — the
  log-probability, in particular, needs only the action noise and the policy's standard
  deviation, never the action mean — so this removes roughly half the per-step kernels while
  computing exactly the same numbers.
- Folding the per-step buffer writes into the compiled step: a further 5%.

What the adversarial review found, all fixed: the fused-CUDA environment launched its kernels
on the legacy default stream instead of the current one, which made its pairing numbers
untrustworthy; priming the observation statistics with that environment stacked one repeated
step, because it returns a preallocated buffer the next step overwrites; and `one_graph` did
not force the compiled rollout it assumes. The review also showed that the campaign's
"copies at goal" column was a sampling artifact, which is why that table now reports the count
both over the final samples and over all of them.

The task also asked directly whether presenting ONLY the on-policy update path would run
faster, on the theory that a branch might block fusion. It does not: a build with the other
path stripped out entirely — verified bitwise identical, then timed in ABBA order — differs by
0.05 to 0.08%, against a 0.01 ms noise floor. The two styles were already separate compiled
functions, so the captured graph contains only the selected path either way.

### Round 3 — sweep hyperparameters across copy groups

Described in its own section below.
"""


def sec_techniques():
    return """## What made it fast (and what did not)

Kept (each row measured in a ledger; see the per-module `progress_and_changes.md` files):

1. Compile the fused per-step function (policy + value + sample + env step + RND bonus)
   with `torch.compile(fullgraph=True)` — 3.4x on the trainer, 9x on the env at 1M envs
   (after restructuring the contact pipeline to be fusable: unrolled 8-candidate loop,
   two-smallest tournament instead of topk, 1-byte neighbor bitmask instead of gathers).
2. Capture whole phases as manual CUDA graphs over static buffers (rollout, update, then
   the entire iteration): 226 -> 45.6 -> 41.5 ms/iter; captured paths bitwise-verified.
3. Capturable + fused Adam with a device-tensor learning rate (required for capture
   correctness, not just speed: a python step counter frozen inside a graph corrupts
   Adam's bias correction).
4. TF32 matmuls (+9%), hoisting the frozen RND target's features out of the minibatch
   loop (+1.6% style B, +6.7% style A), same-input GEMM packing (+1.9% / +2.9%).
5. jax: whole-iteration jit + buffer donation; scan-fused rollout.
6. A single fused CUDA kernel for the env: 24x lower small-batch floor than compiled
   torch (6 us vs 153 us) and 4x at 1M envs (1.68e10 vs 4.0e9 env-steps/s).

Rejected by measurement:

1. Raising inductor's fusion thresholds to force one giant kernel (compile-cost explosion,
   >40 min without producing a kernel).
2. Capturing the EAGER kernel sequence (slower than the per-step-compiled path: replaying
   thousands of tiny kernels is kernel-time bound).
3. `mode="reduce-overhead"` automatic capture (skipped capture silently; manual capture
   with static buffers was both faster and verifiable).

The full extra-step review against the prior joint-replenishment efficiency campaign is in
`extra_step_review.md` (what transferred, what was already covered, and the measurement-
protocol gaps it flagged, several of which were fixed: incremental benchmark JSON writes,
the capturable-Adam coupling, finer env-count ladders near the knee).
"""


def sec_ceiling():
    """How far the implementations sit from what the hardware can do."""
    probe = HERE / ".." / ".." / ".." / "analysis" / "ceiling"
    mf = probe / "data" / "matmul_floor.json"
    cp = probe / "data" / "ceiling_probe.json"
    if not mf.exists():
        return "## How far from the hardware ceiling\n" + pending(
            "ceiling analysis", "the ceiling measurements")
    m = json.loads(mf.read_text())
    c = json.loads(cp.read_text()) if cp.exists() else {}
    peak_tf32 = 417.5e12                       # dense, half the datasheet figure quoted with sparsity
    achieved = max((g["flops_per_s"] for g in c.get("square_gemm", []) if g.get("tf32")),
                   default=None)
    itB = m["iteration_styleB"]
    md = f"""## How far from the hardware ceiling

"An optimisation made it faster" is not the same as "it is now fast". This section asks what the
machine could do in principle, and what fraction of that the implementations reach. Three separate
analyses produced it, cross-checking each other; the full working is in `analysis/ceiling/`.

### The environment

One environment step moves 82 bytes and performs about 250 floating-point operations, counted from
the kernel and then confirmed by compiling it and inspecting the generated instructions. Dividing
the card's memory bandwidth by the bytes gives a ceiling of **48,000 million steps per second**.

| implementation | million steps per second | share of the bandwidth ceiling |
|---|---|---|
| CUDA kernel | 19,300 | 40 percent |
| JAX, many steps per call | 9,920 | 21 percent |
| PyTorch, compiled | 4,166 | 9 percent |

The bandwidth ceiling is, however, the wrong limit for the fastest implementation. The kernel needs
about 12.8 machine instructions per byte it moves, while the processor can only issue about 7.7
instructions per byte at full bandwidth — so it runs out of instruction issue before it runs out of
memory. Its real ceiling is about 28,700 million steps per second, and it achieves **67 percent** of
that.

Two measurements from the project's own record settle which limit binds. Packing the state to cut
memory traffic by 16 percent changed the time by 0.3 percent — if bandwidth were the limit, the time
should have fallen by about 16 percent. Removing arithmetic, by contrast, gained 14 percent. That is
a kernel limited by issuing instructions, not by moving bytes.

### The training computation

One iteration at 128 copies with sixteen updates per batch performs 110.2 thousand million
floating-point operations. Measured against the card's matrix throughput:

| measure | value |
|---|---|
| arithmetic in one iteration | 110.2 GFLOP |
| time for that iteration (PyTorch) | 20.4 ms |
| achieved rate | 5.4 TFLOP/s |
| share of the card's peak matrix rate | 1.3 percent |
"""
    if achieved:
        md += (f"| the largest matrix multiplication this card actually sustains | "
               f"{achieved/1e12:.0f} TFLOP/s ({achieved/peak_tf32*100:.0f} percent of peak) |\n")
    md += f"""| every matrix multiplication in the iteration, timed on its own | {itB['seconds']*1e3:.2f} ms |

A share of 1.3 percent sounds like failure and is not, for two reasons that the analysis makes
precise. First, this algorithm's arithmetic intensity is about 11 operations per byte against a
machine that balances at 106, so **no implementation of it could exceed about 10 percent** of the
peak matrix rate. Second, and more usefully: adding up every matrix multiplication in the iteration,
each timed in isolation, gives {itB['seconds']*1e3:.2f} milliseconds. That is the floor for the
algorithm as written — the time if every other operation were free. The measured 20.4 milliseconds
is **{itB['seconds']*1e3/20.42*100:.0f} percent of that floor**, and the JAX implementation's 13.4
milliseconds is {itB['seconds']*1e3/13.42*100:.0f} percent.

So the honest statement of remaining headroom is not "ninety-nine percent is missing" but "at most
about three times, and only by removing work that is not matrix multiplication".

### Where that places the project

Practitioners generally distinguish five stages: unoptimised; obvious waste removed; the limiting
resource identified and the profile flat; at the hardware limit for the algorithm as written; and
the algorithm itself reformulated to move the limit. On that scale the CUDA environment is at the
fourth stage — its limiting resource is identified by experiment and it runs at two thirds of that
limit — and the trainers are at the third, having gone from 0.8 percent to 2.0 percent of the peak
matrix rate, which is roughly a third of the reachable ceiling for the algorithm as written. Two
reformulations have already been made (batching the copies, fusing the sweep) and one was measured
and declined because it changes the algorithm (a shorter horizon, worth three times).

"""
    return md


def sec_cpu():
    """The same work on ordinary processor cores, for scale."""
    trains = [json.loads(p.read_text()) for p in sorted(RESULTS.glob("*trainbench_cpu*.json"))]
    if not trains:
        return "## The same work on ordinary processor cores\n" + pending(
            "processor comparison", "the processor benchmark files")
    md = """## The same work on ordinary processor cores

To give the graphics-processor numbers a scale, the same code was run on a processor node of the
same cluster (puma01: two sockets of forty cores, 160 hardware threads, 246 GB of memory), inside a
reservation so nothing else was using it.

There are two ways to use many cores, and the difference between them is larger than any
optimisation in this report. **Thread-parallel** means one program holding all the work in one large
array, with each operation split across threads — which requires every thread to finish before the
next operation starts. **Process-parallel** means many independent programs, each with its own share
of the work and no coordination at all.

End-to-end training on processor cores, the best setting of each kind:

| way of using the cores | workers | copies | seconds per iteration | million steps per second | thousand steps per second per copy | hours per million steps per copy |
|---|---|---|---|---|---|---|
"""
    # One row per (way of using the cores, update convention) rather than one per benchmark file:
    # the file count grew past forty as the sweeps were repeated and re-measured, and a table of
    # forty near-duplicate rows hides the four settings that actually differ. The full curves are
    # in the dedicated-node section below.
    # before: 47 rows, several of them the same configuration measured again
    # after:  4 rows, each the highest total its kind reached
    kinds = {}
    for d in trains:
        for r in d["rows"]:
            key = (d["mode"], d.get("style"))
            if key not in kinds or r["env_steps_per_sec"] > kinds[key]["env_steps_per_sec"]:
                kinds[key] = r
    for (mode, style), r in sorted(kinds.items(), key=lambda kv: -kv[1]["env_steps_per_sec"]):
        md += (f"| {mode}, {style} | {r['workers']} | {r['total_copies']:,} | "
               f"{r['sec_per_iteration']:.3f} | {r['env_steps_per_sec']/1e6:.4f} | "
               f"{r['env_steps_per_sec_per_copy']/1e3:,.2f} | "
               f"{H(r['env_steps_per_sec_per_copy'])} |\n")
    md += """
Thread-parallel peaks at a handful of threads and then stops improving: one environment step is
about forty small operations, and the regrouping after each one costs more than the work it
coordinates once the threads are many. The process-parallel form never pays that. The dedicated-node
measurements later in this report put numbers on the difference and settle the choice.

"""
    return md


def sec_choices():
    """Which environment, which trainer, which combination — and why."""
    return """## Which implementation to use

Three practical questions, each answered from the measurements above. The tables that justify these
can be reproduced with `python benchmarks/decide.py`.

**Which environment: the CUDA kernel.** It is fastest at every batch size, and its advantage is
largest for small batches (178 against 6.3 million steps per second at a thousand environments),
where the other implementations spend nearly all their time dispatching work rather than simulating.
All three implementations pass identical exactness checks, so this is purely a speed choice.

**Which trainer: it depends on the copy count, and the answer changes between 128 and 1,024.** At
8 to 128 copies they are close: after the round-four work (below), PyTorch is ahead at 8 copies with
one update per batch and JAX leads by 10 to 22 percent elsewhere, measured the same way on both
sides with each iteration waited for. At 1,024 to 4,096 copies — where this trainer is actually run
— the distance is much larger, JAX by 1.75 to 2.08 times, for a reason that only appears at those
sizes; the last section of this document measures it and says why. The two figures for JAX's lead
predate the round-five PyTorch work in the last section, which closes much of it.

Both compute the same algorithm, and the agreement between them depends on what the card is asked
to do with a matrix multiply:

| matmul precision | worst disagreement across every intermediate quantity |
|---|---|
| exact single precision on both sides | 1.2e-06 |
| the precision both trainers actually ship with | 2.2e-03 |

Both ship with the card's reduced-precision matrix mode — PyTorch asks for it, JAX takes it by
default — so the second row is the one that describes the running trainers, and it is the rounding
that mode is documented to cost rather than a difference between the two implementations.
PyTorch carries the resumable training driver
and the campaign records; both now carry the learning-rate sweep and per-copy progress recording.

**Which combination: keep the environment in the same framework as the trainer.** Substituting the
CUDA kernel, twenty-four times faster on its own, into the PyTorch training loop changes the
iteration by about one percent, because after the optimisation work the environment is a small part
of an iteration. Crossing frameworks is far worse: handing arrays between two runtimes costs about
6.8 milliseconds per environment step against a native step of 0.1 to 0.3 milliseconds, and it
prevents both frameworks from fusing or recording the loop. The fast kernel earns its place in work
that is only environment simulation — generating data, evaluating a fixed policy — not inside this
trainer.

"""


def sec_parity():
    """Feature parity between the two trainers."""
    return """## Feature parity between the two trainers

The learning-rate sweep and per-copy progress recording were first built in PyTorch. They were then
added to JAX, under the requirement that they cost nothing.

| copies | uniform baseline (ms) | with differing rates and per-copy recording (ms) | noise floor (ms) |
|---|---|---|---|
| 8 | 6.90 | 6.86 | 0.06 |
| 128 | 11.93 | 11.93 | 0.10 |
| 512 | 25.16 | 25.18 | 0.33 |
| 2,048 | 81.78 | 81.70 | 1.09 |

Every separated effect — the sweep mechanism, the rates actually differing, the coverage recording —
lands between −0.5 and +0.4 percent, as often negative as positive. This is better than the PyTorch
side's +1.0 percent, and the reason is structural: in JAX the per-copy rates are a constant broadcast
inside the elementwise optimiser update the compiler already emits, so no new operation appears.

The same six properties are tested on both sides: a uniform sweep reproduces the plain run, a
zero-rate group stays exactly frozen while others train, changing one group's rate leaves other
groups unchanged to the last bit, and paired and distinct seeding both behave as documented.

**A measurement lesson worth recording.** The first version of this comparison ran each arm in its
own process, as the other harnesses in this project do. Process-to-process variation alone was 0.68
to 0.72 milliseconds on an 8 to 13 millisecond iteration — larger than every effect being measured —
and increasing the number of timed iterations did not reduce it. Constructing all the arms in one
process and timing them round-robin, with the order reversed on alternate rounds, dropped the floor
to 0.06 to 0.10 milliseconds. Small effects measured with the per-process harnesses elsewhere in this
report rest on a noisier instrument than that.

"""


def sec_round4():
    """Round four: closing the distance between the two trainers."""
    b = {r["n_copies"]: r for r in rows_of(r"trainbench_torch_epoch_minibatch_round4")} \
        if "rows_of" in globals() else {}
    def grab(pat):
        d = newest(pat)
        return {r["n_copies"]: r["sec_per_iteration"] * 1e3 for r in d["rows"]} if d else {}
    nb = grab(r"trainbench_torch_epoch_minibatch_round4")
    na = grab(r"trainbench_torch_full_batch_round4")
    # anchor on the file suffix: a later run wrote *_final_styleA_large.json for the big
    # copy counts, and an unanchored pattern would pick that up instead
    ob = grab(r"trainbench_torch_epoch_minibatch_final_styleB\.json")
    oa = grab(r"trainbench_torch_full_batch_final_styleA\.json")
    jx = newest(r"trainbench_jax_ppo_lfl_sync")
    js = {}
    if jx:
        for r in jx["rows"]:
            js.setdefault(r.get("style"), {})[r["n_copies"]] = (
                r.get("sec_per_iteration") or r.get("sec_per_iteration_median")) * 1e3
    if not nb:
        return "## Round four — closing the distance between the two trainers\n" + pending(
            "round four", "the round-four benchmark files")
    md = """## Round four — closing the distance between the two trainers

Round three left the PyTorch trainer measurably behind the JAX one: level with one update per
batch, but 52 to 70 percent slower with sixteen. Round four attacked that, under the same rule as
every other change — the computation must not change.

| update convention | copies | PyTorch before | PyTorch after | JAX | who leads |
|---|---|---|---|---|---|
"""
    for name, new, old, style in (("one update per batch", na, oa, "full_batch"),
                                  ("sixteen updates per batch", nb, ob, "epoch_minibatch")):
        for c in sorted(new):
            j = js.get(style, {}).get(c)
            lead = ("—" if not j else
                    f"PyTorch by {(j/new[c]-1)*100:.0f} percent" if new[c] < j else
                    f"JAX by {(new[c]/j-1)*100:.0f} percent")
            before = f"{old[c]:.1f}" if c in old else "not measured"
            md += (f"| {name} | {c} | {before} | **{new[c]:.1f}** | {j:.1f} | {lead} |\n" if j
                   else f"| {name} | {c} | {before} | **{new[c]:.1f}** | — | — |\n")
    md += """
*Milliseconds per iteration, both frameworks waiting for each iteration to finish.*

Two changes produced this.

**One flat parameter buffer.** The trainer holds twenty-one parameter tensors per copy. The per-copy
gradient limit had to walk all twenty-one, and so did the optimiser, sixteen times per iteration. The
twenty-one remain separate names, but their storage is now twenty-one windows onto a single buffer, with
the gradients likewise. The limit becomes one reduction and the optimiser one chain: that part of a
minibatch step fell from 1,183 to 122 microseconds. The forward and backward passes also got faster,
which was not the intent — assigning the gradient windows up front removes twenty-one memory
allocations per backward pass.

**Shuffling once per epoch.** The data was previously gathered into position inside each of the
sixteen update steps. Permuting once per epoch into a fixed buffer makes each step a contiguous
slice instead, cutting 144 gather operations per iteration to 36.

**What the profile taught.** The plan handed to this round ranked the gather as a target and did not
mention the gradient limit. A profile taken first showed the opposite: the limit was 31 percent of a
minibatch step and the gather 2 percent. The plan was rewritten from the measurement, and the larger
of the two changes above is the one the plan had omitted.

**An honest loss.** At 512 copies the new arrangement is 1.1 percent *slower*. The buffer is 123
megabytes at that size, the optimiser becomes limited by memory bandwidth, and the extra pass the
per-copy limit needs costs more than the operations it saves. It was kept because it is worth 20 to
29 percent at 8 to 128 copies, which is the range in use, but the regression is recorded rather than
averaged away.

**What remains.** The distance that is left sits in the forward and backward passes themselves,
which are 72 to 84 percent of a minibatch step against 10 to 23 percent for the whole optimiser.
JAX's compiler fuses a small network's gradient into fewer operations than PyTorch's compiler and
automatic differentiation do together. Closing that means writing the loss and its gradient as one
hand-written operation, which is a larger undertaking and was left as a decision rather than begun.

"""
    return md


def throughput_rows(pattern):
    """{(update style, copies): throughput figures} from one benchmark file, {} if absent.

    Both benchmark programs write one row per measured setting; the torch one calls the time
    `sec_per_iteration` and the jax one `sec_per_iteration_median`, so both names are read.
    Every row carries the five figures a throughput table must report.
    before: {"n_copies": 4096, "style": "full_batch", "sec_per_iteration": 0.0868, ...}
    after:  {("full_batch", 4096): {"ms": 86.8, "total": 24.1, "per_copy": 5.9,
                                    "hours": 47.1, "vram": 13.7}}
    """
    d = newest(pattern)
    if not d:
        return {}
    out = {}
    for r in d["rows"]:
        sec = r.get("sec_per_iteration") or r.get("sec_per_iteration_median")
        per_copy = r["env_steps_per_sec"] / r["n_copies"]
        out[(r.get("style"), r["n_copies"])] = {
            "ms": sec * 1e3,
            "total": r["env_steps_per_sec"] / 1e6,          # millions of steps per second
            "per_copy": per_copy / 1e3,                     # thousands per second, one copy
            "hours": 1e6 / (3600.0 * per_copy),             # hours for one copy to reach 1e6 steps
            "vram": r.get("peak_vram_mb", 0.0) / 1024.0,    # gibibytes
        }
    return out


# 8,192 is past the range asked about and was measured only for the changed PyTorch build, to
# see whether the card still holds it; rows with no measurement are simply absent from the tables
LARGE_COPIES = [1024, 2048, 4096, 8192]


def paired_change_rows(pattern):
    """{copies: row} from EVERY paired comparison matching the pattern, newest winning per size.

    Written by `benchmarks/bench_torch_change.py`: both configurations built in one process and
    timed round-robin, the order reversed on alternate rounds, so each round yields one paired
    difference and the verdict is how many rounds favoured the change. One comparison is often
    run over several copy counts in several invocations — the sizes a first pass covered and the
    sizes a later one filled in — so all the matching files are read in filename order and a
    later measurement of a size replaces an earlier one.
    before: two files, one with copies 1024 and 4096, one with 8, 32, 128, 512, 2048, 8192
    after:  one table with all eight sizes
    """
    out = {}
    for p in sorted(q for q in RESULTS.glob("*.json") if re.search(pattern, q.name)):
        for r in json.loads(p.read_text())["rows"]:
            out[r["n_copies"]] = r
    return out


def epilogue_rows(pattern):
    """{arm name: row} from one compiler-generated-multiplication probe, {} if absent."""
    d = newest(pattern)
    return {r["arm"]: r for r in d["rows"]} if d else {}


def large_scale_sources():
    """The measurement files the large-copy-count section reads, as named row tables.

    Three PyTorch states are named because the section covers two rounds of work at these
    sizes: what round five found, what it left, and what round six left. The JAX side is read
    from one file per round — round five measured a revision that a separate line of work was
    changing at the time, and round six re-measured the merged trainer, so both are kept and
    the head-to-head uses the later one.
    """
    return {
        "before A": throughput_rows(r"trainbench_torch_full_batch_base_r5_styleA_large"),
        "before B": throughput_rows(r"trainbench_torch_epoch_minibatch_base_r5_styleB_large"),
        "after A": throughput_rows(r"trainbench_torch_full_batch_after_r5_styleA_large"),
        "after B": throughput_rows(r"trainbench_torch_epoch_minibatch_after_r5_styleB_large"),
        "r6 base A": throughput_rows(r"trainbench_torch_full_batch_r6_base_styleA_large"),
        "r6 base B": throughput_rows(r"trainbench_torch_epoch_minibatch_r6_base_styleB_large"),
        "r6 after A": throughput_rows(r"trainbench_torch_full_batch_r6_after_styleA_large"),
        "r6 after B": throughput_rows(r"trainbench_torch_epoch_minibatch_r6_after_styleB_large"),
        "jax A": throughput_rows(r"trainbench_jax_ppo_base_r5_styleA_large_sync"),
        "jax B": throughput_rows(r"trainbench_jax_ppo_base_r5_styleB_large_sync"),
        # one file per update style: JAX reports peak device memory as a process high-water mark
        # that is never reset, so a process measuring both styles gives the second one the first
        # one's peak. The combined file is the fallback when the split ones are absent.
        "jax r6 A": (throughput_rows(r"trainbench_jax_ppo_r6_base_styleA_large_sync")
                     or throughput_rows(r"trainbench_jax_ppo_r6_base_large_sync")),
        "jax r6 B": (throughput_rows(r"trainbench_jax_ppo_r6_base_styleB_large_sync")
                     or throughput_rows(r"trainbench_jax_ppo_r6_base_large_sync")),
    }


def fig_large_scale():
    """Aggregate and per-copy throughput at 1024-4096 copies, both frameworks, both styles."""
    src = large_scale_sources()
    if not src["before B"] and not src["before A"]:
        return pending("large-copy-count figure", "the 1024-4096 benchmark JSONs")

    series = [("PyTorch before round five", C_TORCH, "--", "o"),
              ("PyTorch after round six", C_TORCH, "-", "o"),
              ("JAX", C_JAX, "-", "^")]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0), dpi=160)
    for row, (style, style_name) in enumerate([("full_batch", "one update per batch"),
                                               ("epoch_minibatch", "sixteen updates per batch")]):
        tag = "A" if style == "full_batch" else "B"
        tables = [src[f"before {tag}"],
                  src[f"r6 after {tag}"] or src[f"r6 base {tag}"] or src[f"after {tag}"],
                  src[f"jax r6 {tag}"] or src[f"jax {tag}"]]
        for col, (key, ylabel) in enumerate(
                [("total", "TOTAL environment steps / second (millions)"),
                 ("per_copy", "per-copy environment steps / second (thousands)")]):
            ax = axes[row][col]
            for (label, color, ls, mk), table in zip(series, tables):
                xs = [c for c in LARGE_COPIES if (style, c) in table]
                if not xs:
                    continue
                ax.plot(xs, [table[(style, c)][key] for c in xs], ls, color=color,
                        linewidth=2, marker=mk, markersize=6, label=label,
                        alpha=0.55 if ls == "--" else 1.0)
            ax.set_xscale("log", base=2)
            ax.set_xticks(LARGE_COPIES)
            ax.set_xticklabels([str(c) for c in LARGE_COPIES])
            ax.set_xlabel("independent training copies")
            ax.set_ylabel(ylabel)
            ax.set_title(style_name, fontsize=10)
            ax.legend(frameon=False, fontsize=8)
            style_ax(ax)
    fig.suptitle("Throughput at the copy counts the trainer is used at (T=128, N=4)", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGS / "large_scale.png")
    plt.close(fig)
    return None


def throughput_table(named_tables, style, copies=None):
    """Render one throughput table: one row per (implementation, copy count).

    named_tables: [(row label, {(style, copies): figures}), ...] in the order to display.
    Every row carries the five figures the throughput convention requires — copies, time for one
    iteration, aggregate rate, per-copy rate, and the hours one copy needs for a million steps —
    plus the peak memory, which is what decides whether a setting fits on the card at all.
    """
    copies = copies or LARGE_COPIES
    # headers are wrapped with line breaks: seven columns of full-length prose would push the
    # table past the right edge of a printed page, where there is no scroll bar to recover it
    md = ["| implementation | copies | milliseconds<br>per iteration | total steps<br>per second"
          "<br>(millions) | steps per second<br>per copy<br>(thousands) | hours per million"
          "<br>steps per copy | peak<br>memory (GB) |",
          "|---|---|---|---|---|---|---|"]
    # bold the best and underline the second best per copy count, on the aggregate rate
    for c in copies:
        present = [(name, t[(style, c)]) for name, t in named_tables if (style, c) in t]
        if not present:
            continue
        ranked = sorted((r["total"] for _, r in present), reverse=True)
        for name, r in present:
            cell = f"{r['total']:.2f}"
            if len(ranked) > 1 and r["total"] == ranked[0]:
                cell = f"**{cell}**"
            elif len(ranked) > 2 and r["total"] == ranked[1]:
                cell = f"<u>{cell}</u>"
            md.append(f"| {name} | {c} | {r['ms']:.1f} | {cell} | {r['per_copy']:.1f} | "
                      f"{r['hours']:.3f} | {r['vram']:.1f} |")
    return "\n".join(md)


def phase_us(pattern):
    """The three phase times and the whole-iteration time of one phase profile, in milliseconds."""
    d = newest(pattern)
    if not d:
        return None
    order = ["rollout", "post-rollout", "update"]
    out = {}
    for key, us in d["phases_us"].items():
        for name in order:
            if key.startswith(name):
                out[name] = us / 1000.0
    out["one graph"] = d["one_graph_us"] / 1000.0
    return out


def sec_large_scale():
    """Round five: the copy counts the trainer is actually used at, 1,024 to 4,096."""
    src = large_scale_sources()
    if not src["before B"] and not src["before A"]:
        return "## Training a thousand to four thousand copies at once\n" + pending(
            "the large-copy-count section", "the 1024-4096 benchmark JSONs")

    md = """## Training a thousand to four thousand copies at once

An earlier section did measure the trainer past a thousand copies, but its figures come from the
first and second optimisation rounds and its JAX figures from the first, and — more importantly —
every optimisation decision recorded anywhere in this document up to that point was taken by
measuring 8 to 128 copies. The trainer is used at 1,024 to 4,096. Two rounds of work re-opened the
question at those sizes; this section reports both, with the current code on both sides: what the
two frameworks cost there, what limits the PyTorch one, and what changed once the limit was
identified.

The short answer is that the two ranges are different problems. At 128 copies the iteration is a
long chain of small device programs and its cost is set by how many there are. At 1,024 copies and
above the same programs each carry eight to thirty-two times as much data, the data no longer fits
in any cache, and the cost is set by how many bytes move between the chip and its memory. An
optimisation that removes device programs helps at 128 copies and does nothing at 4,096; an
optimisation that removes bytes moved helps at 4,096 and does nothing at 128. Both rounds turned on
that distinction, and one change kept in an earlier round on the strength of 8-to-128 measurements
turned out to be a loss at 1,024 and above. Every change below is therefore measured at both ends
of the range before it is kept.

"""
    # the two questions, answered with numbers, before the reader reaches the tables
    a_before = src["before A"]
    a_after = src["r6 after A"] or src["r6 base A"] or src["after A"]
    a_jax = src["jax r6 A"] or src["jax A"]
    key = ("full_batch", 4096)
    if key in a_before and key in a_jax:
        pt = (a_after or a_before)[key]["ms"]
        answer = (
            f"At 4,096 copies with one update per batch, the PyTorch trainer takes "
            f"{a_before[key]['ms']:.0f} milliseconds per iteration as these two rounds found it "
            f"and {pt:.0f} after their changes, against {a_jax[key]['ms']:.0f} for the JAX "
            f"trainer" if a_after else
            f"At 4,096 copies with one update per batch, the PyTorch trainer takes "
            f"{a_before[key]['ms']:.0f} milliseconds per iteration against "
            f"{a_jax[key]['ms']:.0f} for the JAX trainer")
        md += (f"**The two answers in one line.** {answer}. The distance is not made of any one "
               f"slow program — each of PyTorch's runs at 72 to 100 percent of the rate a plain "
               f"copy of memory reaches — but of how many intermediate results have to be "
               f"written to memory and read back between them.\n\n")
    md += """### Where an iteration's time goes as the copy count grows

"""
    ph = {c: phase_us(fr"profile_phases_C{c}(?!.*preround)") for c in (128, 1024, 4096)}
    if all(ph.values()):
        md += ("| phase | 128 copies | 1,024 copies | 4,096 copies |\n|---|---|---|---|\n")
        for name, label in [("rollout", "rollout (128 sequential environment steps)"),
                            ("post-rollout", "post-rollout processing"),
                            ("update", "update (16 minibatch steps)"),
                            ("one graph", "the whole iteration, recorded as one sequence")]:
            md += (f"| {label} | " +
                   " | ".join(f"{ph[c][name]:.2f}" for c in (128, 1024, 4096)) + " |\n")
        md += ("\n*Milliseconds, sixteen updates per batch. The first three lines are the three "
               "phases timed as separately recorded sequences; the last is the shipped "
               "arrangement, which records all three together.*\n\n")
        grow_roll = ph[4096]["rollout"] / ph[128]["rollout"]
        grow_upd = ph[4096]["update"] / ph[128]["update"]
        md += (f"From 128 copies to 4,096 — thirty-two times the work — the rollout grows by a "
               f"factor of {grow_roll:.1f} and the update by a factor of {grow_upd:.1f}. The "
               f"rollout is a chain of about 2,300 small programs whose cost hardly depends on "
               f"how much data each one carries, so adding copies is very nearly free for it. The "
               f"update grows in proportion to the copies, which is what a computation limited by "
               f"memory traffic does.\n\n")

    md += """### The two frameworks at the sizes in use

The two tables below report, for each setting, the time one iteration takes, the aggregate rate
over all copies, the rate a single copy gets, the hours one copy needs to reach a million
environment steps, and the peak memory. Both frameworks wait for every iteration to finish before
timing the next, which is the stricter of the two protocols and the one used everywhere else in
this document.

The rows are three states of the work: "PyTorch before round five" is where the two rounds at
these sizes began, "PyTorch after round six" is where they ended, and "JAX" is
`ppo/jax_ppo/jax_ppo_rnd.py` as it stands after its own fifth round. Every PyTorch and JAX row in
these two tables was measured in one session on the same machine, so the comparison is like for
like — the earlier version of this section reported a JAX revision that a separate line of work
was changing while it was measured, and those figures are superseded here.

"""
    # the last row of each table is whichever PyTorch state is latest, so a round that keeps no
    # change still shows the state it left rather than an empty column
    latest_A = src["r6 after A"] or src["r6 base A"] or src["after A"]
    latest_B = src["r6 after B"] or src["r6 base B"] or src["after B"]
    jax_A = src["jax r6 A"] or src["jax A"]
    jax_B = src["jax r6 B"] or src["jax B"]
    named_A = [("PyTorch before round five", src["before A"]),
               ("PyTorch after round six", latest_A), ("JAX", jax_A)]
    named_B = [("PyTorch before round five", src["before B"]),
               ("PyTorch after round six", latest_B), ("JAX", jax_B)]
    md += "**One update per batch.**\n\n" + throughput_table(named_A, "full_batch") + "\n\n"
    md += ("**Sixteen updates per batch.**\n\n" + throughput_table(named_B, "epoch_minibatch")
           + "\n\n")
    md += ("*Best aggregate rate per copy count in bold, second best underlined. The aggregate "
           "rate rises with the copy count while the rate each individual copy gets falls, so the "
           "hours column is the one that says how long a single training run actually takes. "
           "8,192 copies is past the range this section is about and was measured only for the "
           "changed PyTorch build, to see whether the card still holds it.*\n\n")
    md += ("![Throughput at 1,024 to 4,096 copies](figures/large_scale.png)\n\n")
    # the trade-off the two columns exist to show, stated with the measured numbers
    best = src["r6 after A"] or src["r6 base A"] or src["after A"] or src["before A"]
    if ("full_batch", 1024) in best and ("full_batch", 4096) in best:
        lo, hi = best[("full_batch", 1024)], best[("full_batch", 4096)]
        md += (
            f"The two rate columns move in opposite directions and the choice between copy counts "
            f"depends on which one matters. Going from 1,024 copies to 4,096 with one update per "
            f"batch raises the aggregate rate from {lo['total']:.1f} to {hi['total']:.1f} million "
            f"environment steps per second, a factor of {hi['total']/lo['total']:.2f}, while the "
            f"rate an individual copy gets falls from {lo['per_copy']:.1f} to "
            f"{hi['per_copy']:.1f} thousand per second, a factor of "
            f"{lo['per_copy']/hi['per_copy']:.1f}. In time: one copy reaches ten million "
            f"environment steps, the budget this project's training campaign used, in "
            f"{lo['hours']*10*60:.0f} minutes at 1,024 copies and {hi['hours']*10*60:.0f} minutes "
            f"at 4,096. Four thousand copies is the right setting "
            f"when the science needs many independent runs and the wall time of any one of them "
            f"does not matter; a thousand is the right setting when it does.\n\n")
    md += sec_large_scale_limit()
    md += sec_large_scale_changes()
    md += sec_large_scale_round6()
    md += sec_large_scale_gap(src)
    return md


def sec_large_scale_gap(src):
    """Why the JAX trainer is faster at these copy counts, and what closing it would take."""
    rows = []
    for style, label in (("full_batch", "one update per batch"),
                         ("epoch_minibatch", "sixteen updates per batch")):
        tag = "A" if style == "full_batch" else "B"
        after = (src[f"r6 after {tag}"] or src[f"r6 base {tag}"] or src[f"after {tag}"]
                 or src[f"before {tag}"])
        jax = src[f"jax r6 {tag}"] or src[f"jax {tag}"]
        for c in LARGE_COPIES:
            if (style, c) in after and (style, c) in jax:
                rows.append((label, c, after[(style, c)]["ms"], jax[(style, c)]["ms"]))
    if not rows:
        return ""
    md = """### Why the JAX trainer is still faster at these copy counts

| update convention | copies | PyTorch | JAX | ratio |
|---|---|---|---|---|
"""
    for label, c, pt, jx in rows:
        md += f"| {label} | {c} | {pt:.1f} ms | {jx:.1f} ms | {pt/jx:.2f} |\n"

    # both trainers run the same algorithm on the same shapes, so the counted traffic is the
    # same for both and the ratio to it says how much of each one's cost is avoidable
    sys.path.insert(0, str(BASE / "benchmarks"))
    import count_traffic as ct
    counts = ct.counts("epoch_minibatch")
    floor_ms = sum(v[1] for v in counts.values()) * 4096 / ct.BW * 1e3
    pt4096 = next((p for l, c, p, _ in rows if c == 4096 and "sixteen" in l), None)
    jx4096 = next((j for l, c, _, j in rows if c == 4096 and "sixteen" in l), None)
    floor_md = ""
    if pt4096 and jx4096:
        floor_md = (
            f"\nBoth trainers implement the same algorithm on the same shapes, so the bytes the "
            f"algorithm itself requires are the same for both, and the distance each one sits "
            f"above that figure is comparable:\n\n"
            f"| | milliseconds per iteration | times the counted traffic floor |\n|---|---|---|\n"
            f"| the bytes the algorithm requires, at the card's measured bandwidth | "
            f"{floor_ms:.0f} | 1.00 |\n"
            f"| JAX | {jx4096:.0f} | {jx4096/floor_ms:.2f} |\n"
            f"| PyTorch | {pt4096:.0f} | {pt4096/floor_ms:.2f} |\n\n"
            f"*4,096 copies, sixteen updates per batch. The floor counts every tensor the "
            f"algorithm writes and every later read of it (`benchmarks/count_traffic.py`); it "
            f"does not count intermediates a particular implementation has to materialise, which "
            f"is exactly the quantity the two frameworks differ in.*\n")
    md += """
The reason is not that any PyTorch program is slow. Each is timed on its real shape in the
subsection above and reaches 72 to 100 percent of the rate a plain copy of memory gets. The reason
is that there are more of them, and every program writes its output to memory for the next one to
read.

The two frameworks arrange an iteration differently. The PyTorch trainer records it as a sequence
of separate device programs — a multiplication, then a program that adds the bias and applies the
activation, then the next multiplication — and every intermediate between them is written to
memory and read back. The JAX trainer hands the whole iteration to a compiler that emits fewer
programs, folding chains of element-wise work into the loops that produce and consume them, so a
number of the intermediates PyTorch writes and re-reads are never written at all.
""" + floor_md + """
Round six measured the obvious way to close that difference — having PyTorch's compiler generate
the multiplications so that the bias and the activation fold into them as an epilogue — and the
answer is in the previous subsection. It is not the missing piece. What is left is spread across
every program the iteration issues, and the arrangement of separate library calls is what produces
it.

The peak memory in the tables above points the same way. At 4,096 copies with sixteen updates per
batch the PyTorch trainer holds 12.8 gigabytes after this round (13.8 before it) and the JAX
trainer 9.1. Two deliberate PyTorch choices account for most of the difference: it stores the
frozen target network's features for the whole batch rather than recomputing them in each update
step, and it keeps a second copy of the batch in shuffled order so that each update step is a
contiguous slice rather than its own gather. Both were the right trade at 128 copies and both are
still the right trade on the byte count here, but they are no longer free.

One asymmetry in the comparison, stated so it is not mistaken for part of the gap: the PyTorch
trainer maintains a per-copy map of which maze cells each copy has visited on every iteration,
while the JAX trainer does so only when asked for it, and it was not asked here. That is a
difference in what the two are computing, not in how well they compute it, and it is small — two
integer conversions and a scatter over the stored rows — but it is on the PyTorch side of the
ledger.

"""
    return md


def probe_row(pattern, name_fragment):
    """One measured operation from an operation probe, matched by a fragment of its name."""
    d = newest(pattern)
    if not d:
        return None
    for r in d["rows"]:
        if name_fragment in r["name"]:
            return r
    return None


def sec_large_scale_limit():
    """What sets the cost of an iteration at 4,096 copies, in bytes rather than operations."""
    ref = probe_row(r"probe_update_ops_C4096", "copy one gibibyte")
    adam = probe_row(r"probe_update_ops_C4096", "clip and Adam, compiled")
    adam_eager = probe_row(r"probe_update_ops_C4096", "clip and Adam, eager")
    if not ref:
        return pending("large-copy-count limit", "the operation probe at 4,096 copies")
    sys.path.insert(0, str(BASE / "benchmarks"))
    import count_traffic as ct
    counts = ct.counts("epoch_minibatch")
    C = 4096
    traffic = sum(v[1] for v in counts.values()) * C
    arithmetic = sum(v[0] for v in counts.values()) * C
    measured = (newest(r"profile_phases_C4096(?!.*preround)") or {}).get("one_graph_us", 0) / 1e6
    achieved = traffic / measured if measured else 0.0
    # the isolated matrix-multiply floor: every multiplication of the update stage timed on its
    # own and summed. Reported for the update stage only — for the rollout the same instrument
    # measures a four-row multiplication whose cost is dominated by the fixed cost of issuing it,
    # so its "floor" comes out larger than the rollout actually takes inside a recorded sequence.
    mm = newest(r"matmul_floor_scaled")
    mmfloor = ""
    if mm:
        row = next((r for r in mm["rows"] if r["n_copies"] == 4096), None)
        upd = (newest(r"profile_phases_C4096(?!.*preround)") or {}).get("phases_us", {})
        upd_ms = next((v / 1e3 for k, v in upd.items() if k.startswith("update")), 0.0)
        if row and upd_ms:
            floor_ms = row["update_styleB"]["seconds"] * 1e3
            mmfloor = (
                f"\nThe same conclusion from the other side: every matrix multiplication of the "
                f"update stage, timed on its own and summed, comes to {floor_ms:.0f} milliseconds "
                f"at 4,096 copies, against {upd_ms:.0f} measured for the stage. One third of the "
                f"stage is the multiplications; the other two thirds is moving activations, "
                f"gradients and optimiser state between them.\n")

    layers = [("actor and critic first layer, packed", "actor+critic layer 1"),
              ("actor second layer", "actor layer 2"),
              ("critic second layer", "critic layer 2"),
              ("RND predictor first layer", "predictor layer 1"),
              ("RND predictor second layer", "predictor layer 2"),
              ("RND predictor third layer", "predictor layer 3")]
    md = f"""### What limits the PyTorch trainer at 4,096 copies

The ceiling section earlier in this document analysed the trainer at 128 copies and concluded that
it was limited by the number of separate device programs it issues, not by arithmetic and not by
memory traffic. At 4,096 copies that conclusion no longer holds. Three measurements say so.

**First, arithmetic cannot be the constraint.** One iteration with sixteen updates per batch
performs {arithmetic/1e12:.2f} million million floating-point operations, a figure that follows
from the network shapes and is exact. Counting every tensor the iteration writes and every time a
later operation reads it gives at least {traffic/1e9:.0f} gigabytes of memory traffic, which is a
lower bound because it does not count intermediates the compiler has to materialise. The ratio is
therefore at most {arithmetic/traffic:.1f} operations per byte. The card balances at about 106
operations per byte when its matrix units are used and about 15 when they are not, so this
computation sits far on the memory side of the balance whatever is done to it.

**Second, the card's real bandwidth is {ref['gb_per_s']:,.0f} gigabytes per second**, measured by
copying a gibibyte rather than quoted from the specification, which says 3,900.

**Third, the operations the iteration is made of already run close to that rate.** Each was timed
on its real shape at 4,096 copies:

| operation | time | bytes | rate reached |
|---|---|---|---|
"""
    for label, frag in layers:
        r = probe_row(r"probe_update_ops_C4096", f"bmm alone, {frag}")
        if r:
            md += (f"| multiply, {label} | {r['seconds']*1e6:,.0f} us | {r['bytes']/1e9:.2f} GB | "
                   f"{r['gb_per_s']:,.0f} GB/s |\n")
    for label, frag in [("gradient limit and Adam step, compiled", "clip and Adam, compiled"),
                        ("gather one epoch's rows", "gather the cached target features"),
                        ("plain copy, the reference", "copy one gibibyte")]:
        r = probe_row(r"probe_update_ops_C4096", frag)
        if r:
            md += (f"| {label} | {r['seconds']*1e6:,.0f} us | {r['bytes']/1e9:.2f} GB | "
                   f"{r['gb_per_s']:,.0f} GB/s |\n")
    md += f"""
Nothing in that list is far from the reference. The multiplications reach 72 to 96 percent of the
rate a plain copy gets, the optimiser's pass reaches all of it, and the one operation that is well
below — the gather, which reads rows in a random order — is 3 percent of an iteration. So the
iteration is not slow because any one of its programs is slow. It costs what it costs because of
how many bytes pass through those programs.
{mmfloor}
**The conclusion, and what follows from it.** At 128 copies the trainer was limited by the number
of device programs; at 4,096 copies it is limited by memory traffic, with every program already at
or near the bandwidth the card delivers. A change that removes device programs — which is what
every optimisation in rounds one to four did — cannot help at this size. A change that removes
*passes over memory* can, and the three changes below are all of that kind. For comparison, the
same iteration's arithmetic would take {arithmetic/417.5e12*1e3:.1f} milliseconds if the matrix
units ran at their marketed rate, {arithmetic/417.5e12/measured*100:.1f} percent of the
{measured*1e3:.0f} milliseconds measured; that figure is what the 128-copy analysis reported, and
at this size it says only that the shapes are small, not that there is room in the arithmetic.

"""
    if adam and adam_eager:
        md += (f"One further measurement worth recording: the optimiser's pass over the "
               f"parameters, the moments and the gradients reaches {adam['gb_per_s']:,.0f} "
               f"gigabytes per second compiled and {adam_eager['gb_per_s']:,.0f} uncompiled, so "
               f"compiling it is worth a factor of "
               f"{adam_eager['seconds']/adam['seconds']:.1f} and there is nothing left to win "
               f"inside it.\n\n")
    return md


def sec_large_scale_changes():
    """The three changes this round made, each with the paired measurement that decided it."""
    def ab(pattern):
        d = newest(pattern)
        return d if d else None
    rows = [("aligning the<br>parameter windows", r"ab_round5-align-C1024-styleB"),
            ("adding the bias after<br>the multiplication", r"ab_round5-bias-C1024-styleB"),
            ("writing gradients, and<br>splitting the gradient<br>limit from the Adam step",
             r"ab_round5-gradient-C1024-styleB")]
    got = [(name, ab(pat)) for name, pat in rows]
    # the previous round measured against its own predecessor, which is what identified the
    # layout defect the first change fixes; it belongs here whether or not the per-change
    # comparisons have landed
    prior = [("1,024 copies,<br>one update per batch",
              r"ab_round4-against-predecessor-C1024-styleA"),
             ("1,024 copies,<br>sixteen updates per batch",
              r"ab_round4-against-predecessor-C1024-styleB"),
             ("4,096 copies,<br>sixteen updates per batch",
              r"ab_round4-against-predecessor-C4096-styleB")]
    prior_rows = [(n, ab(p)) for n, p in prior]
    prior_md = ""
    if any(d for _, d in prior_rows):
        prior_md = ("### Round four, measured where the trainer is used\n\n"
                    "The previous round was decided at 8 to 128 copies and recorded a 1.1 percent "
                    "loss at 512 as the single size where it was a loss. Measured against its own "
                    "predecessor at the sizes in use, with both sides pinned to their revisions:\n"
                    "\n| setting | before that round | after it | difference | noise floor |\n"
                    "|---|---|---|---|---|\n")
        for name, d in prior_rows:
            if not d:
                continue
            prior_md += (f"| {name} | {d['a_ms']:.2f} ms | {d['b_ms']:.2f} ms | "
                         f"that round {d['relative_change_percent']*-1:+.1f} percent | "
                         f"{d['noise_floor_ms']:.2f} ms |\n")
        prior_md += ("\nA positive number means the previous round made it slower. The 512-copy "
                     "loss was not an isolated size but the start of a trend, and the first of "
                     "this round's changes is its repair.\n\n")
    if not any(d for _, d in got):
        return prior_md + pending("large-copy-count changes",
                                  "the round-five paired comparisons")
    md = prior_md + """### Round five: what was changed

Three changes, all of them removing passes over memory, none of them changing what the trainer
computes.

**Aligning the parameters.** The previous round put the twenty-one parameter tensors of each copy
into one buffer, as twenty-one windows onto a row of 59,910 numbers. Neither that row length nor
several of the window offsets is a multiple of four, so a given parameter's address for copy *c*
is a multiple of sixteen bytes for almost no *c*, and the matrix library responded by selecting
its kernels that load one number at a time instead of four. A profile of the previous build at
1,024 copies found 18.8 of its 31.1 milliseconds of multiplication time in those unvectorised
kernels. Padding each window and the row to a multiple of four numbers costs ten numbers per copy,
is never read, and restores the vectorised kernels. This also explains a loss the previous round
recorded but could not account for: measured against its own predecessor at 1,024 copies, that
round was 19.2 percent slower with one update per batch and 7.1 percent slower with sixteen —
a real regression at exactly the sizes the trainer is used at, invisible at the sizes it was
tuned on.

**Adding the bias after the multiplication.** Every layer was written as one library call that
multiplies and adds the bias together. There is no batched multiplication that broadcasts a bias,
so that call first writes the expanded bias into the output tensor and then asks the
multiplication to accumulate on top of it; the activation function afterwards reads and writes the
same tensor again. Five passes over the output. Written instead as "multiply, then add the bias
and apply the activation in one expression", the compiler fuses the bias and the activation into
a single pass and the output is touched three times. The result is bitwise identical — the same
sum in the same precision, only computed by different programs — which the equivalence test
records.

**Writing gradients rather than accumulating them, and separating the gradient limit from the
Adam step.** Two changes to the update stage with the same character. First: with a gradient
tensor attached to each parameter, a backward pass *adds* into it, which reads and rewrites the
whole gradient buffer, and the buffer then has to be zeroed before the next step — four passes
over a buffer that holds a gigabyte at 4,096 copies. Asking the automatic-differentiation system
for the gradients instead returns freshly written tensors that nothing has to be added to, and one
call copies them into their windows. Second: the gradient limit and the Adam step were one
compiled function, which the compiler fused into a single program that both reduces and updates;
that program reached 2.4 of the card's 3.5 terabytes per second. Compiled separately, the
reduction reaches 3.2 and the update 3.5.

Each change was then measured on its own against the revision before it, at 1,024 copies with
sixteen updates per batch, both sides built in their own process and run in the order A B B A so
that the spread between two runs of the same side gives the noise floor.

| change | before | after | difference | noise floor |
|---|---|---|---|---|
"""
    for name, d in got:
        if not d:
            continue
        md += (f"| {name} | {d['a_ms']:.2f} ms | {d['b_ms']:.2f} ms | "
               f"{-d['difference_ms']:+.2f} ms ({-d['relative_change_percent']:+.1f} percent) | "
               f"{d['noise_floor_ms']:.2f} ms |\n")
    md += "\n"

    combined = [("4,096 copies,<br>sixteen updates per batch", r"ab_round5-all-C4096-styleB"),
                ("4,096 copies,<br>one update per batch", r"ab_round5-all-C4096-styleA"),
                ("128 copies,<br>sixteen updates per batch", r"ab_round5-all-C128-styleB"),
                ("8 copies,<br>sixteen updates per batch", r"ab_round5-all-C8-styleB")]
    have = [(n, ab(p)) for n, p in combined]
    if any(d for _, d in have):
        md += ("The three together, at the sizes in use and at the small ones the earlier rounds "
               "optimised for:\n\n"
               "| setting | before | after | difference | noise floor |\n|---|---|---|---|---|\n")
        for name, d in have:
            if not d:
                continue
            md += (f"| {name} | {d['a_ms']:.2f} ms | {d['b_ms']:.2f} ms | "
                   f"{-d['difference_ms']:+.2f} ms "
                   f"({-d['relative_change_percent']:+.1f} percent) | "
                   f"{d['noise_floor_ms']:.2f} ms |\n")
        md += "\n"

    # the small-size check: the previous round's loss at 512 copies was found only because the
    # sizes it did NOT optimise for were re-measured, so this round re-measures them too
    small = {"before A": throughput_rows(r"trainbench_torch_full_batch_before_r5_styleA_small"),
             "after A": throughput_rows(r"trainbench_torch_full_batch_after_r5_styleA_small"),
             "before B": throughput_rows(
                 r"trainbench_torch_epoch_minibatch_before_r5_styleB_small"),
             "after B": throughput_rows(
                 r"trainbench_torch_epoch_minibatch_after_r5_styleB_small")}
    if small["after B"]:
        md += ("### Round five: the small sizes, re-measured\n\nThe previous round's loss at 512 copies was "
               "found only because the sizes it had not optimised for were measured afterwards, "
               "so the same check is repeated here in the other direction.\n\n"
               "**Sixteen updates per batch.**\n\n"
               + throughput_table([("PyTorch before", small["before B"]),
                                   ("PyTorch after", small["after B"])],
                                  "epoch_minibatch", [8, 32, 128, 512]) + "\n\n"
               "**One update per batch.**\n\n"
               + throughput_table([("PyTorch before", small["before A"]),
                                   ("PyTorch after", small["after A"])],
                                  "full_batch", [8, 32, 128, 512]) + "\n\n")
    md += """### Round five: whether the three changes changed what the trainer computes

The rule for the round was that they must not. Two of them are exactly neutral and one needs a
sentence.

Adding the bias after the multiplication is **bitwise identical**: the loss and all twenty-one
parameter gradients, computed from the same inputs both ways, agree to zero. Writing the
gradients rather than accumulating them is arithmetically the same sequence of Adam steps; the
only reordering is that the per-copy gradient limit now sums twenty-one tensors' contributions in
the same order as before but in its own program, and one optimiser step from identical inputs
agrees with the previous form to 6.9e-6 relative on parameters that moved 3.0e-4.

Aligning the parameters is the one that needs care. Comparing whole iterations of the two
revisions from one seed gives a difference of 1.08e-3 after a single iteration, which looks
alarming and means nothing: an iteration samples actions, so a difference in the last bits of the
first action sends the two runs down different trajectories, and what is measured afterwards is
divergence rather than error. Asked of the arithmetic in isolation — the same weights read from
an aligned window and from one offset by two numbers, multiplied by the same input — the answer
is exact and in two parts. In full single precision the alignment changes nothing anywhere, in
every layer, to the last bit. In the configuration this trainer actually runs, which enables the
card's reduced-precision matrix units, it changes the two layers whose inner dimension is four,
by about 5e-4 relative. Those two facts together say what happened: when the weights were
misaligned the library could not use the reduced-precision units for those layers and fell back
to full precision, and aligning them lets the setting apply where it previously could not. The
trainer's own configuration documents that setting as costing about 1e-3 of relative rounding,
and it already governs every other multiplication in the program, so the change makes the trainer
more consistent with its own setting rather than quietly less accurate. A run that needs full
single precision throughout has always had to turn that setting off, and with it off the
alignment is exactly neutral.

### Round five: what was considered and not done, with the arithmetic that decided it

Three further ideas were costed against the byte counts above and rejected without being built.
Recording them is the point: two of them look obviously right until the bytes are counted.

1. **Recompute the frozen RND target's features in every update step instead of storing them.**
   This is what the JAX trainer does, and at these sizes it looks like the better trade, because
   storing them spends memory traffic to save arithmetic and memory traffic is the constraint.
   Counted per copy per iteration, storing costs 0.26 megabytes to write the features, 2.1 to
   permute them once per epoch and 1.0 to read them across the sixteen steps: about 3.4 in total.
   Recomputing stores nothing but writes and reads a 256-wide intermediate in every one of the
   sixteen steps, about 6.4 megabytes, plus the target's own weights sixteen times. Storing wins
   by nearly a factor of two.
2. **Process the copies in groups small enough that a group's parameters, moments and gradients
   stay in the 50-megabyte cache across all sixteen update steps.** That would remove fifteen
   sixteenths of the parameter traffic, which is about a third of an iteration — by far the
   largest remaining saving. A copy's working set is about 1.65 megabytes, so a group that fits
   the cache holds about thirty copies, and a program with thirty pieces of work cannot fill the
   card's 132 processing blocks. The two requirements cannot both be met at this network size.
3. **Hold the optimiser's two moments in a narrower number format.** It removes two of the eight
   passes the optimiser makes, about 5 percent of an update step. It changes what the trainer
   computes, so it belongs in a round whose rule permits that, with its own equivalence gate,
   rather than in this one.

"""
    return md


def paired_table(rows, label, copies=None):
    """One paired round-by-round comparison rendered as a table, newest measurement per size."""
    copies = copies or sorted(rows)
    # the two long headers are wrapped: a table wider than about a hundred characters runs
    # off the right edge of a printed page, where there is no scroll bar to recover it
    md = [f"| copies | before | {label} | change | rounds<br>favouring it | "
          "spread within<br>a version |", "|---|---|---|---|---|---|"]
    for c in copies:
        r = rows.get(c)
        if not r:
            continue
        md.append(f"| {c} | {r['median_sec']['off']*1e3:.2f} ms | "
                  f"{r['median_sec']['on']*1e3:.2f} ms | "
                  f"{r['change_percent']:+.1f} percent | "
                  f"{r['rounds_favouring_on']} of {r['rounds']} | "
                  f"{r['noise_floor_sec']*1e3:.2f} ms |")
    return "\n".join(md)


def sec_large_scale_round6():
    """Round six: what the JAX side's findings and method transferred to the PyTorch trainer."""
    grad_B = paired_change_rows(r"torch_change_gradient_buffer_epoch_minibatch_sync")
    grad_A = paired_change_rows(r"torch_change_gradient_buffer_full_batch_sync")
    # two invocations cover different sizes: 1,024 and 4,096 by knob name, 8 to 512 through the
    # multi-knob form because those arms also have to name the copy-major layout
    fused = paired_change_rows(r"torch_change_fuse_copy_and_limit_epoch_minibatch_sync")
    fused.update(paired_change_rows(r"torch_change_set_epoch_minibatch_sync_small"))
    versus = paired_change_rows(r"torch_change_set_epoch_minibatch_sync_fused_vs_nobuffer")
    layout = paired_change_rows(r"torch_change_set_epoch_minibatch_sync_layout")
    gather = newest(r"ab_r6-gather-out-C4096-styleB")
    epi = epilogue_rows(r"probe_epilogue_C4096_pertrainer") or epilogue_rows(r"probe_epilogue_C4096")
    # anchored on the extension: the same probe also writes a parameter-major file, and that one
    # holds only the no-buffer programs, because the buffer form does not exist in that layout
    prog = newest(r"probe_gradient_form_C4096\.json")
    prog_pm = newest(r"probe_gradient_form_C4096_parameter_major\.json")
    if not grad_B:
        return pending("the round-six subsection", "the round-six paired comparisons")

    md = """### Round six: what transferred from the other framework, and two more passes removed

The JAX trainer's fifth round finished after this document's round-five section was written, so
its findings had never been read from the PyTorch side. This subsection reports what transferred,
what did not, and the changes that followed.

#### What transferred, and what did not

**The measurement method transferred.** That round found that comparing two medians against the
largest spread a single version shows between rounds called a real three-percent effect "noise",
because both versions drift together within a round. The two versions are timed in the same round
on the same machine, so the difference belongs round by round, and a version that wins every round
is faster whatever the between-round spread is. Every comparison below uses that statistic, through
a harness (`benchmarks/bench_torch_change.py`) that builds both versions in ONE process and times
them round-robin with the order reversed on alternate rounds. The older spread is reported beside
it, so the two rules can be compared on the same data.

**The direction transferred.** Both of that round's gains came from making the compiler emit fewer,
longer-running programs, which at these sizes means fewer round trips through memory for the same
work.

**Its two kept changes did not, and that is a result.** Both are unroll factors on loops. The
PyTorch trainer has no loop to unroll: its sixteen update steps are already emitted one after
another into a single recorded sequence of device programs, so there is no loop for a compiler to
emit more copies of. The JAX round is also worth nothing at these sizes on its own side — its
trainer measures within a percent of what it did before that round at 1,024 to 4,096 copies, and
the gains it reports are at 8 to 128 copies, where an iteration's cost is the number of programs
it issues. The regime split cuts both ways.

#### Where the update stage's time goes at 4,096 copies

Every program of one iteration, named and summed by kind. This is what chose the changes below.

| part of the update stage | time | share |
|---|---|---|
| matrix multiplications | 62.72 ms | 37.6% |
| the Adam step | 31.94 ms | 19.1% |
| copying the gradients into the flat buffer | 16.70 ms | 10.0% |
| shuffling the batch once per epoch | 6.42 ms | 3.8% |
| the per-copy gradient limit | 4.58 ms | 2.7% |
| everything else (bias, activations, and their gradients) | 44.66 ms | 26.7% |

*Sixteen updates per batch, 4,096 copies, the three stages timed without graph capture so the
profiler can name each program (`benchmarks/profile_kernels.py`).*

The Adam step and the gradient limit already run at the full 3,540 gigabytes per second a plain
copy of memory reaches on this card, so there is nothing left inside them. The copy does not: it
reaches 1,878, because its destination is a strided window of the buffer rather than a contiguous
tensor. That is the largest single removable item, and it is what round five's own change created.

"""
    md += """#### The changes, each measured against the revision before it

"""
    md += ("**Reading the gradients where the backward pass wrote them**, instead of copying "
           "them into one buffer so that the gradient limit and the Adam step can each be a "
           "single program over one contiguous array. The price is twenty-one programs per "
           "optimizer pass instead of one; the measurement is whether that costs more than the "
           "copy it removes.\n\n")
    md += paired_table(grad_B, "reading them in place") + "\n\n"
    md += ("*Sixteen updates per batch. A negative change is faster.*\n\n"
           "**The sign reverses, and where it reverses is the whole point of this section.** At "
           "8 to 128 copies an iteration costs what it costs because of how many device programs "
           "it issues, and forty-two programs per update step where there were two is a bad trade for one copy "
           "of a buffer that is only 2 to 31 megabytes there. At 512 and above the same copy is "
           "123 megabytes to a gigabyte and the programs are large enough that their number "
           "stops mattering. The trainer therefore chooses between the two forms by copy count "
           "(`production_config`), which is the same shape of answer the previous round arrived "
           "at from the other direction.\n\n")
    if grad_A:
        md += ("The same change with one update per batch, where the iteration has one update "
               "step rather than sixteen and so pays for the copy once rather than sixteen "
               "times:\n\n" + paired_table(grad_A, "reading them in place") + "\n\n")
    if fused:
        md += ("A third form was measured because it removes a different pass: keep the buffer, "
               "but sum the squared gradients in the same program that copies them into it, so "
               "the buffer is never read a second time for the gradient limit. This is the form "
               "the trainer uses below the crossover, where the buffer stays.\n\n"
               + paired_table(fused, "with the limit fused into the copy") + "\n\n")
    if versus:
        md += ("The two new forms against each other, so the choice between them is measured "
               "rather than inferred from their separate comparisons:\n\n"
               + paired_table(versus, "no buffer at all") + "\n\n")
    if layout:
        md += (
            "**One contiguous block per parameter, instead of one buffer row per copy.** The "
            "round-four buffer holds [copies, parameters]: the layout a single-program optimizer "
            "needs, because a copy then owns a contiguous row and one kernel can walk the whole "
            "array applying that copy's rate. With the optimizer now twenty-one programs, that "
            "is the wrong layout — a parameter's window of it is strided across copies, and a "
            "pass over a strided window reaches 2,022 gigabytes per second against 3,588 for the "
            "same pass over a contiguous tensor. Holding one block per parameter instead makes "
            "every window a multiplication or the optimizer touches contiguous, and keeps every "
            "copy on a sixteen-byte boundary by padding each block's per-copy length to a "
            "multiple of four numbers. Only four of the twenty-one need that padding, and none "
            "of them is a multiplication operand.\n\n"
            + paired_table(layout, "one block per parameter") + "\n\n"
            "*Both sides read the gradients where they were written, so the only difference is "
            "the layout. The same expressions run over the same numbers at different addresses, "
            "and on the processor the two train to bitwise equal parameters. On the card they do "
            "not, for the reason the change exists: the multiplication library picks its kernel "
            "partly from the operand's layout, so a contiguous weight and a strided one go "
            "through different kernels, which sum the same products in a different order. One "
            "iteration from identical inputs puts the gradients 4.5e-08 apart, against a largest gradient of 8.6e-01, and the parameters after a step 7.8e-11 apart in relative terms.*\n\n")
    if gather:
        md += (
            "**Writing the shuffled batch straight into its buffer.** Once per epoch the whole "
            "batch is permuted into a second buffer so that each of the four update steps is a "
            "contiguous slice of it. Written as `buffer.copy_(t.gather(...))` the permutation "
            "allocates a whole second copy of the batch and then copies it across. Written as "
            "`torch.gather(t, 1, ix, out=buffer)` it does not.\n\n"
            "| measurement | before | after |\n|---|---|---|\n"
            "| device-to-device copying in the update stage, 4,096 copies | 2.86 ms | absent |\n"
            f"| one whole iteration, 4,096 copies, separate processes | {gather['a_ms']:.2f} ms |"
            f" {gather['b_ms']:.2f} ms |\n\n"
            f"*The second row is at the resolution limit of the cross-process harness — its "
            f"noise floor here is {gather['noise_floor_ms']:.2f} ms — which is why the program "
            f"that disappears is quoted as well. This change cannot be a knob, so it cannot be "
            f"measured by the paired harness, which compares two configurations of one build.*"
            "\n\n")
    if prog:
        md += ("Why the gradient forms differ, program by program at 4,096 copies "
               "(`benchmarks/probe_gradient_form.py`):\n\n"
               "| program | form | time | bytes | rate |\n|---|---|---|---|---|\n")
        for r in prog["rows"]:
            # the probe names each program as "what it does, which form it belongs to"; the two
            # halves become two columns so neither cell runs off the edge of a printed page
            what, _, form = r["name"].rpartition(", ")
            md += (f"| {what} | {form} | {r['microseconds']:,.0f} us | "
                   f"{r['bytes']/1e9:.2f} GB | {r['gb_per_s']:,.0f} GB/s |\n")
        if prog_pm:
            for r in prog_pm["rows"]:
                what, _, form = r["name"].rpartition(", ")
                md += (f"| {what} | {form}, one block per parameter | "
                       f"{r['microseconds']:,.0f} us | {r['bytes']/1e9:.2f} GB | "
                       f"{r['gb_per_s']:,.0f} GB/s |\n")
        md += ("\nPer minibatch step the buffer form's three programs come to "
               f"{prog['per_step_us']['buffer form']:,.0f} microseconds and the other form's two "
               f"to {prog['per_step_us']['no-buffer form']:,.0f}, so "
               f"{prog['per_step_us']['buffer form'] - prog['per_step_us']['no-buffer form']:,.0f} "
               f"microseconds a step and "
               f"{(prog['per_step_us']['buffer form'] - prog['per_step_us']['no-buffer form']) * 16 / 1000:.1f}"
               " milliseconds over the sixteen steps of an iteration — which is what the "
               "whole-iteration comparison above measures. The decomposition also says what the "
               "change gives back: reading the gradients in place means the Adam step walks "
               "twenty-one windows instead of one contiguous array, and the layout change "
               "recovers about a third of that.\n\n")
    if epi:
        md += """#### Tried and not kept: letting the compiler generate the multiplications

Round five measured this and set it aside with the note that "the compiler's chosen kernels set
`ALLOW_TF32=False`", so adopting it would silently stop using the card's reduced-precision matrix
units. This round found the reason, and it is a size rule in the compiler's own template
heuristics: a generated multiplication is allowed those units only when it has at least sixteen
rows AND the smaller of its two inner dimensions is at least 512. Every multiplication in this
trainer has an inner dimension of 4, 64, 128 or 256, so the rule refuses all of them — while the
multiplication library is under no such rule and does use the units at an inner dimension of four,
which is what round five's alignment measurement showed when it found those layers moving by 5e-4
as the setting was switched on.

So it was measured with the library backend removed, so that a generated kernel is used even where
the library's is faster — which is the point, because leaving both backends offered means the
library's kernel wins the selection on nearly every shape and no epilogue fuses at all — and again
with the size rule replaced by the configuration's own answer.

| form | update stage at 4,096 copies | against the shipped form | rounds faster | loss differs by |
|---|---|---|---|---|
"""
        for name, label in [("library", "the library's multiplication, as shipped"),
                            ("generated", "the compiler's, both backends offered"),
                            ("generated_triton", "the compiler's, library backend removed"),
                            ("generated_tf32", "the same, with the matrix units allowed")]:
            r = epi.get(name)
            if not r:
                continue
            diff = ("—" if name == "library"
                    else f"{r.get('relative_loss_difference', 0.0):.1e} relative")
            md += (f"| {label} | {r['median_us']/1000:.2f} ms | "
                   f"{r['change_percent_against_library']:+.1f} percent | "
                   f"{r['rounds_faster_than_library']} of {r['rounds']} | {diff} |\n")
        md += ("\nThe generated kernels are not close. The selection log has the library's "
               "multiplication at 0.071 milliseconds against the best generated candidate's "
               "0.078 on the first shape it tries, and the backward pass's transposed shapes are "
               "worse; the passes an epilogue would remove cannot pay for that. Switching the "
               "matrix units back on changes nothing, so round five's reason for setting this "
               "aside was a real observation about a form that was not going to pay anyway.\n\n"
               "**A note on how this was measured, because the first attempt measured nothing.** "
               "The first version of the probe built ONE trainer and swapped four compiled "
               "versions of its loss onto it, each compiled inside a context that set the "
               "compiler's options. All four came out bitwise identical and within 0.1 percent "
               "of each other in time — one form measured four times, not four forms agreeing. "
               "The compiler caches its work against the function being compiled and against the "
               "backend the wrapper carries, and a setting applied through a surrounding context "
               "is part of neither. The probe now gives each arm its own trainer and passes the "
               "settings as options rather than around them, and it says so out loud when two "
               "arms agree to zero. The accuracy line it already printed is what caught it.\n\n")
    # the three kept changes together, against the revision the round started from
    combined = [("4,096 copies,<br>sixteen updates per batch", r"ab_r6-all-C4096-styleB"),
                ("4,096 copies,<br>one update per batch", r"ab_r6-all-C4096-styleA"),
                ("128 copies,<br>sixteen updates per batch", r"ab_r6-all-C128-styleB"),
                ("8 copies,<br>sixteen updates per batch", r"ab_r6-all-C8-styleB")]
    have = [(n, newest(p)) for n, p in combined]
    if any(d for _, d in have):
        md += ("#### The three together, against the revision the round started from\n\n"
               "| setting | before | after | difference | noise floor |\n|---|---|---|---|---|\n")
        for name, d in have:
            if not d:
                continue
            md += (f"| {name} | {d['a_ms']:.2f} ms | {d['b_ms']:.2f} ms | "
                   f"{-d['difference_ms']:+.2f} ms "
                   f"({-d['relative_change_percent']:+.1f} percent) | "
                   f"{d['noise_floor_ms']:.2f} ms |\n")
        md += ("\n*Both sides built in their own process and run in the order A B B A, so the "
               "spread between two runs of the same side is the noise floor. At 8 and 128 copies "
               "the trainer keeps the gradient buffer, so only the shuffle change applies "
               "there.*\n\n")

    # the same check the previous round's loss at 512 copies made necessary: the sizes this round
    # did NOT optimise for, measured before and after rather than assumed unchanged
    small = {"before A": throughput_rows(r"trainbench_torch_full_batch_r6_before_styleA_small"),
             "after A": throughput_rows(r"trainbench_torch_full_batch_r6_after_styleA_small"),
             "before B": throughput_rows(
                 r"trainbench_torch_epoch_minibatch_r6_before_styleB_small"),
             "after B": throughput_rows(
                 r"trainbench_torch_epoch_minibatch_r6_after_styleB_small")}
    if small["after B"]:
        md += ("#### The small sizes, re-measured\n\nThe trainer picks the gradient form from the "
               "copy count, so 8 to 512 copies keep the buffer, with the gradient limit summed "
               "inside the copy, and the shuffle change. The whole round is measured there "
               "anyway, because that is how the previous round's loss at 512 copies was found. "
               "No size is slower.\n\n"
               "**Sixteen updates per batch.**\n\n"
               + throughput_table([("before round six", small["before B"]),
                                   ("after round six", small["after B"])],
                                  "epoch_minibatch", [8, 32, 128, 512]) + "\n\n"
               "**One update per batch.**\n\n"
               + throughput_table([("before round six", small["before A"]),
                                   ("after round six", small["after A"])],
                                  "full_batch", [8, 32, 128, 512]) + "\n\n")
    md += """#### Whether round six changed what the trainer computes

One change is exactly neutral and two are not, and the two that are not are the same change to the
same sum, so there is one thing to establish rather than three.

Writing the shuffled batch straight into its buffer moves the same rows in the same order to the
same place; nothing about the arithmetic differs, and the capture test still reports the recorded
iteration as bitwise equal to the uncaptured one.

Holding one contiguous block per parameter instead of one row per copy runs the same expressions
over the same numbers at different addresses. On the processor that is bitwise identical, and a
test trains the trainer for two iterations in each layout and requires exact equality on every
parameter. On the card it is not, because the multiplication library picks its kernel partly from
the operand layout: a contiguous weight and a strided one go through different kernels, which sum
the same products in a different order. That is the change working rather than a caveat around it,
and one iteration from identical inputs puts the gradients 4.5e-08 apart, against a largest
gradient of 8.6e-01, and the parameters after a step 7.8e-11 apart in relative terms.

Reading the gradients where the backward pass wrote them changes the order in which the per-copy
gradient limit adds its squares: one contiguous reduction over a row of 59,920 numbers becomes
twenty-one sub-reductions summed. That is a reassociation of a float32 sum, so it cannot be
bitwise neutral, and the question is whether it is the same quantity. It is: recomputed in double
precision from the same gradients, the two norms agree to **1.4e-16 relative**. In single
precision one optimizer step from byte-identical inputs moves the parameters by 3.0e-4 and the two
forms land 3.0e-8 apart, which is 2.9e-7 of the largest parameter.

| gate | result |
|---|---|
| the two gradient forms, one step from identical inputs | 3.0e-08 absolute, 2.9e-07 relative |
| the same two gradient norms, recomputed in double precision | 1.4e-16 relative |
| the two buffer layouts, two iterations of training on the processor | bitwise equal |
| the two buffer layouts, one iteration on the card | gradients 4.5e-08 of a largest 8.6e-01; parameters 7.8e-11 relative |
| the recorded iteration against the uncaptured one, both update conventions | 0.000e+00 |
| the annealed rate reaches the recorded graph; a zero-rate group stays frozen | 0.000e+00 |
| six learning-rate-sweep gates | all pass |
| every parameter, moment and gradient window on a sixteen-byte<br>boundary, in both layouts, checked on more than one copy | pass |

#### The largest inefficiency left, measured but not attempted

The kernel profile names one item that neither round went after, and it is not in the update
stage. Of the rollout's 15.3 milliseconds at 4,096 copies, **11.45 are matrix multiplications
running at about 850 gigabytes per second** — about a quarter of the rate the update stage's
multiplications reach, and the counted floor for the whole rollout is 2.8 milliseconds.

The cause is structural rather than a missing optimisation. The rollout is 128 sequential
environment steps, and each step multiplies **four rows per copy** — one per environment — against
that copy's entire actor weights. Across 4,096 copies those weights are 75.6 megabytes, they do
not fit the card's 50-megabyte cache, and they are therefore re-read from memory on all 128 steps
to serve almost no arithmetic each time.

| | rollout at 4,096 copies |
|---|---|
| measured | 15.33 ms |
| of which matrix multiplications | 11.45 ms |
| the rate those multiplications reach | ~850 GB/s |
| the bytes the rollout requires, at the card's measured bandwidth | 2.80 ms |

Two shapes of answer exist and neither is small. Processing the copies in groups whose weights fit
the cache would make the re-reads come from there, at the price of doubling the program count in a
stage that is already partly bound by it. Generating the multiplications for these four-row shapes
instead of calling the library's, whose kernels are plainly not tuned for four rows, is the other
— and this round's measurement of that route on the update stage's shapes, where it lost by 42
percent, says nothing about these, which are a different problem. Recorded with its measurement
rather than attempted at the end of a round.

"""
    return md


def sec_repro():
    return """## Reproduction

1. Environments: `pointmaze/{torch_env,cuda_env,jax_env}`; shared exact-physics spec and
   fixtures in `pointmaze/common/` (`physics_spec.md`, checker
   `common/code/check_against_fixtures.py --impl torch|cuda|jax`).
2. Trainers: `ppo/torch_ppo/torch_ppo_rnd.py`, `ppo/jax_ppo/jax_ppo_rnd.py`; tests under
   each `tests/`.
3. Benchmarks: `benchmarks/bench_env_step*.py`, `bench_train*.py`,
   `profile_breakdown.py`; every JSON in `benchmarks/results/` carries the git hash.
   For the last section: `profile_kernels.py` (device time per individual program),
   `probe_update_ops.py` (each operation of the update stage against the bandwidth a plain copy
   reaches), `matmul_floor_scaled.py` (every matrix multiplication timed on its own, at any copy
   count), `count_traffic.py` (arithmetic and bytes per iteration, no device needed),
   `compare_revisions.py` (two revisions run from one seed, worst parameter difference), and
   `run_round5_remote.sh` (the batch, run from the machine itself). `bench_train.py`,
   `profile_phases.py` and `profile_kernels.py` all take `--rev` so a past revision can be
   measured by the same harness in the same session.
4. Final campaign: `train_runs/run_final.py` (resumable, one process per copy count);
   run folder `train_runs/2026-08-15-02-56_final_...` with `experiment_background.md`.
5. This report: `report/.../code/make_report.py` regenerates `report.md` and all figures.
"""


def sec_clean_node():
    """The dedicated-node processor comparison, and the best setup within 4,096 copies.

    Both sections are produced by the standalone report's `cpu_sections` module. Importing that
    module rather than copying its code keeps ONE definition of these numbers, so re-running a
    measurement updates both documents and they cannot come to disagree.
    """
    # the module writes its figures into its own report by default; redirect them into this one
    cpu_sections.FIGS = FIGS
    # every figure that module draws, found by name rather than listed by hand: it is edited by
    # whoever is measuring the processor nodes, and a figure added there but not drawn here leaves
    # the page referencing an image file that does not exist
    for fig_name in sorted(n for n in dir(cpu_sections) if n.startswith("fig_")):
        note = getattr(cpu_sections, fig_name)()
        if note:
            PENDING.append(note)
    md = cpu_sections.sec_cpu() + "\n" + cpu_sections.sec_best()
    # this document numbers no sections, and already carries a processor section about the other
    # node, so the source document's numbering goes and the two headings get names that do not
    # collide with it (a repeated heading would collapse two entries of the contents table)
    # before: "## 5. The same work on ordinary processor cores" / "### 5.1 Why this comparison is here"
    # after:  "## End-to-end training on a dedicated processor node" / "### Why this comparison is here"
    md = md.replace("## 5. The same work on ordinary processor cores",
                    "## End-to-end training on a dedicated processor node")
    md = md.replace("## 6. The best setup on each platform",
                    "## The best setup on each platform, at 4,096 copies or fewer")
    return re.sub(r"^### \d+\.\d+ ", "### ", md, flags=re.M)


def main():
    FIGS.mkdir(exist_ok=True)
    fig_env_throughput()
    fig_copy_scaling()
    fig_campaign_curves()
    fig_sweep_curves()
    fig_sweep_scaling()
    fig_sweep_vs_separate()
    fig_uniform_vs_sweep()
    import cpu_node_comparison as cnc
    cnc.fig_cpu_node_comparison()
    fig_large_scale()
    import section_times as st
    body = [sec_correctness(), sec_module1(), sec_module2(), sec_module3(), sec_copies(),
            sec_profile(), sec_before_after(), sec_campaign(), sec_sweep(), sec_sweep_scaling(),
            sec_uniform_vs_sweep(), sec_rounds(), sec_techniques(), sec_repro(),
            # sections added later in the project go at the end, in the order they were added
            sec_ceiling(), sec_cpu(), sec_choices(), sec_parity(), sec_round4(),
            sec_clean_node(), cnc.sec_cpu_node_comparison(), sec_large_scale()]
    sections = st.split_sections("\n".join(body))
    manifest = st.stamp({k: v for k, v in sections.items() if k != "(title and introduction)"})
    order = [ln[3:].strip() for ln in "\n".join(body).splitlines() if ln.startswith("## ")]
    md = "\n".join([sec_overview(), st.table_of_contents(order, manifest)] + body)
    (REPORT / "report.md").write_text(md)
    print(f"wrote {REPORT/'report.md'} and figures")
    for p in PENDING:
        print("PENDING:", p)


if __name__ == "__main__":
    main()
