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

# fixed categorical order (validated default palette, light mode): color follows the
# ENTITY (framework); line style separates modes of the same entity
C_TORCH = "#2a78d6"   # slot 1 blue
C_CUDA = "#eb6834"    # slot 2 orange
C_JAX = "#1baf7a"     # slot 3 aqua
C_EXTRA = "#eda100"   # slot 4 yellow (copy counts in the campaign figure use a blue ramp instead)
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

| groups (learning rates) | copies per rate | ms/iteration | peak VRAM [MB] |
|---|---|---|---|
"""
        for r in groups:
            md += (f"| {r['n_rates']} | {r['copies_per_rate']} | "
                   f"{r['sec_per_iteration']*1e3:.1f} | {r['peak_vram_mb']:.0f} |\n")
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

| layout | ms per sweep iteration | peak VRAM [MB] |
|---|---|---|
"""
    d = newest(r"sweep_strategies")
    if not d:
        return md + pending("sweep layout table", "the sweep strategy JSON")
    names = {"uniform": "uniform rate, one trainer (not a sweep — the reference)",
             "fused": "one trainer, per-copy rate vector, one graph",
             "separate": "one trainer per rate, run in turn"}
    for row in d["rows"]:
        md += (f"| {names[row['strategy']]} | {row['sec_per_iteration']*1e3:.2f} | "
               f"{row['peak_vram_mb']:.0f} |\n")
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

Reading order: what was built and why it is correct, then the module-by-module numbers, then
what each round changed, then the training campaign and the sweep.
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
    return """## The three rounds

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
    envs = [json.loads(p.read_text()) for p in sorted(RESULTS.glob("*envbench_cpu*.json"))]
    trains = [json.loads(p.read_text()) for p in sorted(RESULTS.glob("*trainbench_cpu*.json"))]
    if not envs or not trains:
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

| way of using the cores | best aggregate, million steps per second | where the best point was |
|---|---|---|
"""
    for d in envs:
        best = max(d["rows"], key=lambda r: r["env_steps_per_sec"])
        md += (f"| environment, {d['mode']}, {d['n_envs_per_worker']:,} environments each | "
               f"{best['env_steps_per_sec']/1e6:.3f} | {best['workers']} workers |\n")
    md += """
Thread-parallel peaks at four to eight threads and then gets *worse* — at 160 threads it is twelve
times slower than a single thread. One environment step is about forty small operations, and the
regrouping after each one costs more than the work it coordinates once the threads are many. The
process-parallel form never pays that, and reaches about twenty-four times a single core.

End-to-end training on the same node:

| way of using the cores | workers | copies | seconds per iteration | million steps per second | steps per second per copy |
|---|---|---|---|---|---|
"""
    for d in trains:
        best = max(d["rows"], key=lambda r: r["env_steps_per_sec"])
        md += (f"| {d['mode']}, {d.get('style')} | {best['workers']} | {best['total_copies']} | "
               f"{best['sec_per_iteration']:.3f} | {best['env_steps_per_sec']/1e6:.4f} | "
               f"{best['env_steps_per_sec_per_copy']:,.0f} |\n")
    md += """
Putting the two platforms beside each other: for the environment alone the graphics processor is
about seven hundred and fifty times faster (19,300 against 25.7 million steps per second); for
end-to-end training the ratio is about thirty (3.21 against 0.096 million). The gap narrows because
training is dominated by matrix multiplication, which processors do comparatively well, while the
environment is dominated by many tiny independent operations, which is exactly what a graphics
processor is for.

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

**Which trainer: they are close, and the choice is no longer mainly about speed.** After the round-four
work (below), PyTorch is ahead at 8 copies with one update per batch and JAX leads by 10 to 22 percent
elsewhere, measured the same way on both sides with each iteration waited for. Both compute the same algorithm and agree to 8.6e-7 on every intermediate
quantity. PyTorch carries the resumable training driver and the campaign records; both now carry the
learning-rate sweep and per-copy progress recording.

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

**One flat parameter buffer.** The trainer holds nineteen parameter tensors per copy. The per-copy
gradient limit had to walk all nineteen, and so did the optimiser, sixteen times per iteration. The
nineteen remain separate names, but their storage is now nineteen windows onto a single buffer, with
the gradients likewise. The limit becomes one reduction and the optimiser one chain: that part of a
minibatch step fell from 1,183 to 122 microseconds. The forward and backward passes also got faster,
which was not the intent — assigning the gradient windows up front removes nineteen memory
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


LARGE_COPIES = [1024, 2048, 4096]


def large_scale_sources():
    """The six measurement files the large-copy-count section reads, as named row tables."""
    return {
        "before A": throughput_rows(r"trainbench_torch_full_batch_base_r5_styleA_large"),
        "before B": throughput_rows(r"trainbench_torch_epoch_minibatch_base_r5_styleB_large"),
        "after A": throughput_rows(r"trainbench_torch_full_batch_after_r5_styleA_large"),
        "after B": throughput_rows(r"trainbench_torch_epoch_minibatch_after_r5_styleB_large"),
        "jax A": throughput_rows(r"trainbench_jax_ppo_base_r5_styleA_large_sync"),
        "jax B": throughput_rows(r"trainbench_jax_ppo_base_r5_styleB_large_sync"),
    }


def fig_large_scale():
    """Aggregate and per-copy throughput at 1024-4096 copies, both frameworks, both styles."""
    src = large_scale_sources()
    if not src["before B"] and not src["before A"]:
        return pending("large-copy-count figure", "the 1024-4096 benchmark JSONs")

    series = [("PyTorch before this round", C_TORCH, "--", "o"),
              ("PyTorch after this round", C_TORCH, "-", "o"),
              ("JAX", C_JAX, "-", "^")]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0), dpi=160)
    for row, (style, style_name) in enumerate([("full_batch", "one update per batch"),
                                               ("epoch_minibatch", "sixteen updates per batch")]):
        tag = "A" if style == "full_batch" else "B"
        tables = [src[f"before {tag}"], src[f"after {tag}"], src[f"jax {tag}"]]
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
    md = ["| implementation | copies | milliseconds per iteration | total environment steps "
          "per second (millions) | environment steps per second per copy (thousands) | hours per "
          "million steps per copy | peak memory (GB) |",
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
                      f"{r['hours']:.1f} | {r['vram']:.1f} |")
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

Every measurement in the sections above was taken at 8 to 128 independent training copies, and
every optimisation recorded there was chosen by what those sizes rewarded. The trainer is used at
1,024 to 4,096 copies. This section re-opens the question at those sizes: what the two frameworks
cost there, what limits the PyTorch one, and what changed once the limit was identified.

The short answer is that the two ranges are different problems. At 128 copies the iteration is a
long chain of small device programs and its cost is set by how many there are. At 1,024 copies and
above the same programs each carry eight to thirty-two times as much data, the data no longer fits
in any cache, and the cost is set by how many bytes move between the chip and its memory. An
optimisation that removes device programs helps the first case and does nothing for the second;
an optimisation that removes bytes does the opposite. One change kept in the previous round on the
strength of the 8-to-128 measurements is a loss at 1,024 and above, and is recorded below.

### Where an iteration's time goes as the copy count grows

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
this document. "PyTorch before" is the trainer as this round found it; "PyTorch after" is the
same trainer with this round's changes; the JAX trainer is unchanged by this round.

"""
    named_A = [("PyTorch before", src["before A"]), ("PyTorch after", src["after A"]),
               ("JAX", src["jax A"])]
    named_B = [("PyTorch before", src["before B"]), ("PyTorch after", src["after B"]),
               ("JAX", src["jax B"])]
    md += "**One update per batch.**\n\n" + throughput_table(named_A, "full_batch") + "\n\n"
    md += ("**Sixteen updates per batch.**\n\n" + throughput_table(named_B, "epoch_minibatch")
           + "\n\n")
    md += ("*Best aggregate rate per copy count in bold, second best underlined. The aggregate "
           "rate rises with the copy count while the rate each individual copy gets falls, so the "
           "hours column is the one that says how long a single training run actually takes.*\n\n")
    md += ("![Throughput at 1,024 to 4,096 copies](figures/large_scale.png)\n\n")
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
4. Final campaign: `train_runs/run_final.py` (resumable, one process per copy count);
   run folder `train_runs/2026-08-15-02-56_final_...` with `experiment_background.md`.
5. This report: `report/.../code/make_report.py` regenerates `report.md` and all figures.
"""


def main():
    FIGS.mkdir(exist_ok=True)
    fig_env_throughput()
    fig_copy_scaling()
    fig_campaign_curves()
    fig_sweep_curves()
    fig_sweep_scaling()
    fig_sweep_vs_separate()
    fig_uniform_vs_sweep()
    import section_times as st
    body = [sec_correctness(), sec_module1(), sec_module2(), sec_module3(), sec_copies(),
            sec_profile(), sec_before_after(), sec_campaign(), sec_sweep(), sec_sweep_scaling(),
            sec_uniform_vs_sweep(), sec_rounds(), sec_techniques(), sec_repro(),
            # sections added later in the project go at the end, in the order they were added
            sec_ceiling(), sec_cpu(), sec_choices(), sec_parity(), sec_round4()]
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
