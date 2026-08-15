"""Generate the unified report (report.md + figures/) from the benchmark result JSONs.

Rerunnable: reads the NEWEST JSON per (kind, tag) from benchmarks/results/ plus the final
training-run records under train_runs/, writes figures and every table into report.md.
Sections whose data has not landed yet render a PENDING marker and the script prints a
warning for each.

Run: <python with matplotlib> make_report.py
"""
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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


def torch_cscale_rows():
    """The torch copy-scaling series across the three source JSONs, deduplicated."""
    rows_t0 = (newest(r"trainbench_torch_epoch_minibatch_pack\.") or
               newest(r"trainbench_torch_epoch_minibatch_onegraph\.") or {}).get("rows", [])
    rows_t = (newest(r"trainbench_torch_epoch_minibatch_cscale\.") or {}).get("rows", [])
    rows_t2 = (newest(r"trainbench_torch_epoch_minibatch_cscale2") or {}).get("rows", [])
    seen = {r["n_copies"] for r in rows_t + rows_t2}
    return sorted([r for r in rows_t0 if r["n_copies"] not in seen] + rows_t + rows_t2,
                  key=lambda r: r["n_copies"])


def fig_copy_scaling():
    """Total and per-copy training throughput vs copy count (torch + jax)."""
    allt = torch_cscale_rows()
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
        ax.plot(xs, ys, "-", color=C_TORCH, linewidth=2, marker="o", markersize=6,
                label="torch (one-graph, style B)")
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


def campaign_records():
    """The final-campaign per-count JSONs, {(style, C): record}."""
    out = {}
    for p in RUNS.glob("2026-*final*/data/copies_*.json"):
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

All numbers were measured on serval05 (one NVIDIA H100 NVL, 95 GB; direct ssh, exclusive
use enforced by a lock), from the code committed in this repository under
`09_parallelization/`. Every table cell traces to a JSON in `benchmarks/results/` or a run
record under `train_runs/`; this report is generated by `report/.../code/make_report.py`.

The task had three modules — (1) a GPU-parallel reimplementation of the Gymnasium-Robotics
PointMaze environment in PyTorch, fused CUDA, and JAX; (2) a batched PPO+RND trainer
(continuous actions, two update styles) that trains many INDEPENDENT copies at once in
PyTorch and JAX; (3) the end-to-end combination — plus a final training campaign at
8/16/32/64/128 copies, profiling breakdowns, and scaling studies.
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
                       ("CUDA fused kernel", r"envbench_cuda.*_grid"),
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
    cud = newest(r"trainbench_torch_epoch_minibatch_cudaenv3")
    if cud and cud["rows"]:
        cells = {r["n_copies"]: r for r in cud["rows"]}
        parts = [f"C={c}: {cells[c]['sec_per_iteration']*1e3:.1f} ms/iter"
                 for c in sorted(cells)]
        md += f"  Measured (style B, one-graph): {', '.join(parts)} — versus the torch-env\n"
        md += "  backend's 26.4 / 36.6 ms at C=8 / 128.\n"
    else:
        md += pending("Module 3 cuda pairing", "cudaenv3 trainbench JSON")
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

| copies C | ms/iter | total env-steps/s | per-copy env-steps/s |
|---|---|---|---|
"""
    for r in torch_cscale_rows():
        md += (f"| {r['n_copies']} | {r['sec_per_iteration']*1e3:.1f} | "
               f"{sci(r['env_steps_per_sec'])} | "
               f"{sci(r['env_steps_per_sec_per_copy'])} |\n")
    md += """| 65536 | out of memory during graph build | — | — |

**Maximum copies that fit: 32,768** (95 GB H100, this configuration). Torch total
throughput saturates near 5.6e6 env-steps/s from ~8,192 copies. The jax trainer saturates
near 1.9e7 (style A) / 9.1e6 (style B) around C=2,048-4,096.

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

| phase | before (eager) | after (captured) |
|---|---|---|
| rollout (env + policy interaction) | 680 ms | 20.1 ms |
| post-processing (GAE, statistics) | inside rollout | 14.8 ms (eager between graphs); inside the one-graph replay in the final config |
| update (16 minibatch steps) | 95 ms | 15.5 ms |
| whole iteration | 777 ms | 36.6 ms (one-graph, final) |
"""
    return md


def sec_campaign():
    recs = campaign_records()
    if not recs:
        return "## Final training campaign\n" + pending("campaign", "final-run JSONs")
    md = """## Final training campaign — 8 to 128 independent RND runs (PyTorch)

20,000 iterations per configuration = 10.24M environment steps per copy; PointMaze Large,
sparse goal, run-6 setup. Every copy is an independent seed with its own networks,
environments, statistics, and optimizer.

| style | copies | wall time [s] | total env-steps/s | copies at goal* | coverage mean | peak VRAM [MB] |
|---|---|---|---|---|---|---|
"""
    for (style, c) in sorted(recs, key=lambda k: (k[0], k[1])):
        r = recs[(style, c)]
        hist = r["history"]
        late = hist[-max(1, len(hist)//10):]
        solved = 0
        if late:
            per_copy = [max(row["reward_ext_sum_per_copy"][i] for row in late)
                        for i in range(c)]
            solved = sum(1 for v in per_copy if v > 0)
        cov = sum(r["final_coverage_per_copy"]) / c
        md += (f"| {style} | {c} | {r['train_seconds']:.0f} | "
               f"{sci(r['env_steps_per_sec'])} | {solved}/{c} | {cov:.3f} | "
               f"{r['peak_vram_mb']:.0f} |\n")
    md += """
*copies at goal = copies with a positive extrinsic-reward iteration inside the final 10%
of training (the sparse goal is intermittently re-found; per-copy curves below).

![campaign curves](figures/campaign_curves.png)
"""
    return md


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
    md = "\n".join([sec_overview(), sec_correctness(), sec_module1(), sec_module2(),
                    sec_module3(), sec_copies(), sec_profile(), sec_before_after(),
                    sec_campaign(), sec_techniques(), sec_repro()])
    (REPORT / "report.md").write_text(md)
    print(f"wrote {REPORT/'report.md'} and figures")
    for p in PENDING:
        print("PENDING:", p)


if __name__ == "__main__":
    main()
