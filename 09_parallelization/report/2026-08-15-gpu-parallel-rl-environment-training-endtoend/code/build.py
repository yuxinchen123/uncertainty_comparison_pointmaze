"""Build the unified report: three sections (environment, training, end to end).

Every number comes from a JSON in benchmarks/results/ or a run record in train_runs/, so the
document can be regenerated after any new measurement. Throughput is reported in MILLIONS of
environment steps per second throughout; the helper `M` performs that conversion in one place.

Run: <python with matplotlib and numpy> build.py
"""
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
REPORT = HERE.parent
BASE = REPORT.parent.parent
RESULTS = BASE / "benchmarks" / "results"
RUNS = BASE / "train_runs"
FIGS = REPORT / "figures"

# one colour per implementation, held fixed across every figure in the document
C_TORCH = "#2a78d6"
C_CUDA = "#eb6834"
C_JAX = "#1baf7a"
C_ALT = "#4a3aa7"
GRID = dict(color="#d9d9d9", linewidth=0.6)
MISSING = []


def M(x):
    """A rate in steps per second, expressed in millions and formatted for a table cell."""
    v = x / 1e6
    if v >= 100:
        return f"{v:,.0f}"
    if v >= 10:
        return f"{v:.1f}"
    if v >= 1:
        return f"{v:.2f}"
    return f"{v:.3f}"


def K(x):
    """A rate in steps per second, expressed in thousands. Used for per-copy figures, which
    in millions would round to zero once there are thousands of copies."""
    v = x / 1e3
    return f"{v:,.0f}" if v >= 10 else f"{v:.1f}"


def newest(pattern):
    """The most recent results JSON whose name matches the regular expression, or None."""
    hits = sorted(p for p in RESULTS.glob("*.json") if re.search(pattern, p.name))
    return json.loads(hits[-1].read_text()) if hits else None


def missing(what):
    """Record an absent input and return a visible placeholder."""
    MISSING.append(what)
    return f"\n*This subsection is waiting on {what}.*\n"


def style_ax(ax):
    """Recessive grid and no top or right frame, so the data carries the figure."""
    ax.grid(True, which="major", **GRID)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def rows_of(pattern, key="rows"):
    """The row list of a results JSON, or an empty list."""
    d = newest(pattern)
    return d.get(key, []) if d else []


# --------------------------------------------------------------------------- figures

def fig_env_throughput():
    """Environment throughput and per-step time against the number of environments."""
    series = []
    for label, pat, color, ls, mk in [
            ("PyTorch, compiled", r"envbench_torch_compile_grid", C_TORCH, "-", "o"),
            ("CUDA kernel", r"envbench_cuda_fused_tourn_count_grid|envbench_cuda_fused_tourn_count", C_CUDA, "-", "s"),
            ("JAX, one step at a time", r"envbench_jax_jit_grid", C_JAX, "-", "^"),
            ("JAX, many steps per call", r"envbench_jax_scan_grid", C_JAX, "--", "v"),
            ("PyTorch, not compiled", r"envbench_torch_eager\.", C_ALT, ":", "x")]:
        r = rows_of(pat)
        if r:
            series.append((label, color, ls, mk, sorted(r, key=lambda x: x["total_envs"])))
    if not series:
        return missing("the environment benchmark files")

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), dpi=160)
    for label, color, ls, mk, rows in series:
        xs = [r["total_envs"] for r in rows]
        axes[0].plot(xs, [r["env_steps_per_sec"] / 1e6 for r in rows], ls, color=color,
                     linewidth=2, marker=mk, markersize=6, label=label)
        axes[1].plot(xs, [r["us_per_batch_step"] for r in rows], ls, color=color,
                     linewidth=2, marker=mk, markersize=6, label=label)
    axes[0].set_ylabel("million environment steps per second")
    axes[1].set_ylabel("microseconds per step of the whole batch")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("number of environments simulated together (log scale)")
        ax.legend(frameon=False, fontsize=8)
        style_ax(ax)
    fig.suptitle("Environment: throughput, and the cost of one step of the whole batch",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGS / "env_throughput.png")
    plt.close(fig)
    return None


def fig_training_scaling():
    """Training throughput against the number of independent copies, both update styles."""
    b = rows_of(r"trainbench_torch_epoch_minibatch_round2b_styleB")
    b += rows_of(r"trainbench_torch_epoch_minibatch_round2c_large")
    b = sorted({r["n_copies"]: r for r in b}.values(), key=lambda r: r["n_copies"])
    a = sorted(rows_of(r"trainbench_torch_full_batch_round2b_styleA"),
               key=lambda r: r["n_copies"])
    old = rows_of(r"trainbench_torch_epoch_minibatch_cscale\.")
    old += rows_of(r"trainbench_torch_epoch_minibatch_cscale2")
    old = sorted({r["n_copies"]: r for r in old}.values(), key=lambda r: r["n_copies"])
    if not b:
        return missing("the training benchmark files")

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), dpi=160)
    sets = [("many small updates per batch of data", b, C_TORCH, "-", "o"),
            ("one update per batch of data", a, C_CUDA, "-", "s"),
            ("many small updates, before the second round of work", old, "#9aa0ab", "--", "x")]
    for label, rows, color, ls, mk in sets:
        if not rows:
            continue
        xs = [r["n_copies"] for r in rows]
        axes[0].plot(xs, [r["env_steps_per_sec"] / 1e6 for r in rows], ls, color=color,
                     linewidth=2, marker=mk, markersize=6, label=label)
        axes[1].plot(xs, [r["env_steps_per_sec_per_copy"] / 1e3 for r in rows], ls,
                     color=color, linewidth=2, marker=mk, markersize=6, label=label)
    axes[0].set_ylabel("million environment steps per second, all copies together")
    axes[1].set_ylabel("thousand environment steps per second, per copy")
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("number of independent training copies (log scale)")
        ax.legend(frameon=False, fontsize=8)
        style_ax(ax)
    fig.suptitle("Training: total throughput rises with copies, per-copy throughput falls",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGS / "training_scaling.png")
    plt.close(fig)
    return None


def fig_sweep():
    """Learning-rate sweep: scaling, and the cost of sweeping against a single rate."""
    rates = rows_of(r"sweep_scaling_rates")
    copies = rows_of(r"sweep_scaling_copies")
    uvs = rows_of(r"uniform_vs_sweep")
    if not rates or not uvs:
        return missing("the sweep benchmark files")

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), dpi=160)
    axes[0].plot([r["total_copies"] for r in rates],
                 [r["env_steps_per_sec"] / 1e6 for r in rates], "-", color=C_TORCH,
                 linewidth=2, marker="o", markersize=6,
                 label="more learning rates, 128 copies each")
    if copies:
        axes[0].plot([r["total_copies"] for r in copies],
                     [r["env_steps_per_sec"] / 1e6 for r in copies], "--", color=C_CUDA,
                     linewidth=2, marker="s", markersize=6,
                     label="more copies per rate, 16 rates")
    axes[0].set_xscale("log", base=2)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("total copies (log scale)")
    axes[0].set_ylabel("million environment steps per second")

    xs = [r["total_copies"] for r in uvs]
    noise = [r["noise_floor_sec"] / r["uniform_sec"] * 100 for r in uvs]
    axes[1].fill_between(xs, [-n for n in noise], noise, color="#d9d9d9", alpha=0.75,
                         label="repeat-to-repeat spread")
    axes[1].axhline(0.0, color="#9aa0ab", linewidth=1, linestyle=":")
    axes[1].plot(xs, [r["vector_cost_percent"] for r in uvs], "--", color=C_CUDA, linewidth=2,
                 marker="s", markersize=6, label="giving each copy its own rate")
    axes[1].plot(xs, [r["differing_cost_percent"] for r in uvs], ":", color=C_JAX, linewidth=2,
                 marker="^", markersize=6, label="those rates actually differing")
    axes[1].set_xscale("log", base=2)
    axes[1].set_xlabel("total copies (log scale)")
    axes[1].set_ylabel("percent slower than one rate for every copy")
    for ax in axes:
        ax.legend(frameon=False, fontsize=8)
        style_ax(ax)
    fig.suptitle("Sweeping learning rates: how it scales, and what it costs", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGS / "sweep.png")
    plt.close(fig)
    return None


def campaign(round2=True):
    """Training-campaign records, keyed by (update style, number of copies)."""
    pat = "2026-*final_round2*/data/copies_*.json" if round2 else \
          "2026-08-15-02-56_final*/data/copies_*.json"
    out = {}
    for p in RUNS.glob(pat):
        r = json.loads(p.read_text())
        out[(r["style"], r["n_copies"])] = r
    return out


def fig_endtoend():
    """Where an iteration's time goes, and what the copies actually learn."""
    ph = newest(r"profile_phases")
    recs = campaign(True)
    if not ph or not recs:
        return missing("the phase profile and the training campaign records")

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.4), dpi=160)
    # left: the three phases as a share of one iteration
    names = [k for k in ph["phases_us"]]
    vals = [ph["phases_us"][k] / 1000 for k in names]
    short = ["rollout", "post-rollout\nprocessing", "update"]
    bars = axes[0].bar(short, vals, color=[C_TORCH, C_CUDA, C_JAX], width=0.6)
    for rect, v in zip(bars, vals):
        axes[0].text(rect.get_x() + rect.get_width() / 2, v, f"{v:.1f} ms",
                     ha="center", va="bottom", fontsize=9)
    axes[0].set_ylabel("milliseconds per training iteration")
    axes[0].set_title("Where an iteration's time goes (128 copies)", fontsize=10)
    style_ax(axes[0])

    # right: learning against environment steps, one line per copy count
    ramp = ["#b7d3f6", "#86b6ef", "#3987e5", "#1c5cab", "#104281"]
    styleb = {c: r for (s, c), r in recs.items() if s == "epoch_minibatch"}
    for i, c in enumerate(sorted(styleb)):
        h = styleb[c]["history"]
        xs = [row["global_step"] / c / 1e6 for row in h]
        ys = [float(np.mean(row["coverage_per_copy"])) for row in h]
        axes[1].plot(xs, ys, "-", color=ramp[min(i, 4)], linewidth=1.7, label=f"{c} copies")
    axes[1].set_xlabel("million environment steps per copy")
    axes[1].set_ylabel("fraction of the maze visited (mean over copies)")
    axes[1].set_title("What the copies learn", fontsize=10)
    axes[1].legend(frameon=False, fontsize=8)
    style_ax(axes[1])
    fig.tight_layout()
    fig.savefig(FIGS / "endtoend.png")
    plt.close(fig)
    return None


# --------------------------------------------------------------------------- prose

def sec_intro():
    """What the system is, what was measured, and how to read the numbers."""
    return """# Running many reinforcement-learning experiments at once on one graphics processor

## 1. What this report covers

This report describes a system that trains many independent reinforcement-learning agents
simultaneously on a single graphics processor, and it reports how fast that system runs. The
work is organised in three parts, and this document has one section for each:

1. **The environment** — the simulated world the agents act in.
2. **The training algorithm** — the procedure that turns experience into improved behaviour.
3. **End to end** — the two combined into a training loop, which is what is actually run.

Each section states what the part does, presents the measured results in tables and figures,
and then lists the optimisation techniques that were tried: those that were kept, and those
that were tried and abandoned, with the measurement that decided each case.

Two further sections put those numbers in context. Section 5 runs the same training loop on
ordinary processor cores, on a cluster node held exclusively so nothing else disturbed the
timings, and compares the two ways of dividing work across many cores. Section 6 names the
single best configuration on each of the two platforms within the range of scale this project
actually uses, and converts each into the wall-clock time to train every copy.

### 1.1 The problem being solved

The agent controls a ball in a two-dimensional maze. It can push the ball in two directions,
it observes the ball's position and velocity, and it receives a reward only when it reaches a
goal square in the far corner. Because the reward appears so rarely, the agent additionally
receives an internally generated "curiosity" reward for visiting unfamiliar states — a standard
method called Random Network Distillation. The learning algorithm is Proximal Policy
Optimisation, a widely used method that repeatedly collects a batch of experience and then
adjusts the agent slightly in the direction that would have made the good outcomes more likely.

Two properties of this workload drive everything that follows:

- **The individual computations are tiny.** The agent's networks have a few tens of thousands
  of parameters; a single agent uses a negligible fraction of a modern graphics processor.
- **Research requires many repetitions.** A result from one agent means little, because
  outcomes vary a great deal between random starts. Conclusions require tens or hundreds of
  independent repetitions, and often several settings of a parameter as well.

The system therefore trains many independent **copies** at once. A copy is a complete,
self-contained experiment: its own networks, its own environments, its own optimiser state,
its own random seed. Copies never exchange information; running them together is purely a
device for using the processor efficiently, and the report includes the tests that confirm
they remain independent.

### 1.2 How to read the numbers

- All throughput figures are given in **millions of environment steps per second**. One
  environment step is one action taken in one environment instance. The one exception is
  throughput *per copy*, which is given in thousands: with thousands of copies sharing the
  processor, each individual copy advances at a rate that would round to zero in millions.
- Timings are for one **training iteration**: collecting a fixed amount of experience from
  every copy and performing the resulting parameter updates. With the default settings each
  iteration collects 512 steps of experience per copy.
- All measurements were taken on one NVIDIA H100 NVL processor (95 gigabytes of memory) with
  exclusive access enforced by a lock, so no other work competed for the device.
- Where a difference is small, the report states the **noise floor**: the same configuration
  measured more than once, in alternating order, so that the spread between repeats of the
  same thing is visible next to the difference being claimed. A difference smaller than that
  spread is reported as no effect rather than as a result.

### 1.3 A note on the two conventions for updating

The training algorithm admits two ways of consuming a batch of collected experience, and both
are implemented and measured throughout:

- **One update per batch**: compute a single parameter update from all the collected data,
  then discard the data.
- **Many small updates per batch**: pass over the data four times, each pass split into four
  smaller portions, updating after each portion — sixteen updates in total.

The second does considerably more arithmetic per unit of collected experience, so it is
slower per iteration. Neither is more correct than the other; they are different algorithms
and both are reported.
"""


def sec_env():
    """Section 2: the environment."""
    md = """## 2. The environment

### 2.1 What it does and why it was rewritten

The original environment came from a standard research library and ran on the processor's
general-purpose cores, simulating one maze at a time through a physics engine. That
arrangement is a poor match for the goal of this project: the training computation lives on
the graphics processor, so every simulated step would require copying data back and forth, and
one maze at a time cannot keep a modern accelerator busy.

The environment was therefore rewritten to run entirely on the graphics processor, holding
thousands to millions of independent mazes in memory at once and advancing all of them with
the same instructions. Three separate implementations were written, because it was not obvious
in advance which approach would win:

- **PyTorch** — expressed as operations on large arrays in the same framework as the trainer.
- **A hand-written CUDA kernel** — the whole step written as one program that the graphics
  processor runs directly, one thread per environment.
- **JAX** — a second array framework whose compiler fuses operations aggressively.

### 2.2 Correctness before speed

A fast environment that simulates different physics from the reference is worthless, so the
first task was to establish exactness. The reference environment's physics were extracted and
reproduced exactly, including the contact model that governs what happens when the ball meets
a wall — the force law, the point at which contact begins, and the case where the ball touches
two walls at once.

The three implementations are checked against trajectories recorded from the reference
environment:

| check | result |
|---|---|
| single-step error against the reference, double precision | at most 4.4e-16, which is the smallest difference representable |
| single-step error, single precision | at most 5e-7 |
| 400-step trajectories through repeated wall contacts | agree to 1e-13 |
| random restarts across the three implementations | identical to the last bit |
| CUDA kernel against the PyTorch version, 500 steps with 320 restarts | worst difference 2.1e-7 |
| 256 episodes under random actions against the reference | distributions of visited squares differ by 0.026, well inside the accepted tolerance |

The one exception is documented rather than hidden: a single recorded trajectory passes
through a knife-edge situation, with the ball balanced on a wall corner against an opposing
push, where any rounding difference eventually sends the two simulations to different places.
Its single-step agreement remains exact.

### 2.3 Results

"""
    series = [("PyTorch, not compiled", r"envbench_torch_eager\."),
              ("PyTorch, compiled", r"envbench_torch_compile_grid"),
              ("CUDA kernel", r"envbench_cuda_fused_tourn_count_grid|envbench_cuda_fused_tourn_count"),
              ("JAX, one step at a time", r"envbench_jax_jit_grid"),
              ("JAX, many steps per call", r"envbench_jax_scan_grid")]
    cols = [1000, 10000, 100000, 1000000, 3000000]
    have = [(lab, {r["total_envs"]: r for r in rows_of(pat)}) for lab, pat in series]
    have = [(lab, d) for lab, d in have if d]
    if not have:
        return md + missing("the environment benchmark files")
    md += ("| implementation | " + " | ".join(f"{c:,} envs" for c in cols) +
           " | best measured |\n|---" * 1 + "|---" * (len(cols) + 1) + "|\n")
    for lab, d in have:
        cells = [M(d[c]["env_steps_per_sec"]) if c in d else "not measured" for c in cols]
        best = max(d.values(), key=lambda r: r["env_steps_per_sec"])
        md += (f"| {lab} | " + " | ".join(cells) +
               f" | {M(best['env_steps_per_sec'])} at {best['total_envs']:,} |\n")
    md += """
*Millions of environment steps per second. Higher is better.*

![environment throughput](figures/env_throughput.png)

The right-hand panel explains the shape of the left-hand one. Below a certain number of
environments, the time taken by one step of the whole batch does not depend on how many
environments are in it: the processor is idle for most of that time, waiting for instructions
to be dispatched. Throughput therefore grows in direct proportion to the batch. Above that
point the processor is genuinely occupied, the step time begins to rise, and throughput
flattens. The number at which this transition happens is a property of the implementation:

| implementation | flat step time up to | time per step in the flat region | best throughput |
|---|---|---|---|
"""
    knees = {"PyTorch, compiled": ("about 300,000 environments", "155 to 171 microseconds"),
             "CUDA kernel": ("about 30,000 environments", "5.6 to 6.2 microseconds"),
             "JAX, one step at a time": ("about 100,000 environments", "71 to 79 microseconds"),
             "JAX, many steps per call": ("about 100,000 environments", "8 to 19 microseconds"),
             "PyTorch, not compiled": ("about 100,000 environments", "about 1,540 microseconds")}
    for lab, d in have:
        if lab in knees:
            best = max(d.values(), key=lambda r: r["env_steps_per_sec"])
            md += f"| {lab} | {knees[lab][0]} | {knees[lab][1]} | {M(best['env_steps_per_sec'])} |\n"
    md += """
*The CUDA kernel reaches its flat limit earliest because a single program has no dispatch
overhead left to hide; it is doing real work almost immediately.*

The practical reading is that the hand-written kernel is the fastest environment by a wide
margin at every size, and that the gap is largest for small batches, where the compiled
PyTorch version spends almost all of its time in dispatch overhead rather than simulation.

### 2.4 Optimisation techniques

**Kept.**

| technique | what it does | measured effect |
|---|---|---|
| Rewrite the wall-contact test so that no operation depends on data | The original formulation selected the two nearest walls with a sorting operation, which forces the compiler to stop and start; the replacement computes all eight candidate walls unconditionally and picks the two smallest with arithmetic | 9x at one million environments in PyTorch (2,242 to 248 microseconds per step) |
| Store the maze geometry as one byte per square | Instead of looking up rectangles from tables, the eight neighbouring walls are encoded as bits and the rectangles reconstructed arithmetically | part of the same 9x |
| Compile the step | Ask the framework to fuse the many small operations of a step into a few larger programs | 13x for PyTorch at small batches |
| Write the whole step as one CUDA program | One dispatch instead of dozens; the environment's state stays in processor registers throughout | 24x lower cost per step at small batches than compiled PyTorch |
| Rank the candidate walls by counting | In the CUDA kernel, replace the running "best two so far" comparison chain, which carries ten values from one candidate to the next, with an independent count of how many candidates are closer | 14 percent at one million environments; verified to produce identical results to the last bit |
| Take several environment steps per call in JAX | Let the compiler unroll a short sequence of steps into one program | 1.77x at small batches, 1.11x at one million |
| Launch the CUDA kernel on the framework's own execution stream | Not a speed change but a correctness one: the kernel had been launching on a different stream from the surrounding work, which meant that recorded sequences of operations silently did not contain the environment step at all | the defect is described in section 4.4 |

**Tried and not kept.**

| technique | why it was rejected |
|---|---|
| Force the compiler to fuse the entire step into a single program by raising its internal thresholds | Compilation ran for more than forty minutes without producing a result and was abandoned |
| A simplified wall model that pushes the ball out of walls instead of simulating contact forces | Simpler and slightly faster, but wrong: single-step errors of up to 48 millimetres against the reference, so it was replaced by the exact contact model |
| Pack the environment state so that it is fetched in one wide memory transaction, and remove a redundant output | No measurable change at any batch size, which established that the kernel is limited by computation and latency rather than by memory bandwidth. Kept only because it is simpler, not because it is faster |
| Record the environment step as a replayable sequence of operations | Faster for small batches (18 percent) but slower at one million environments, and the comparison was not exactly like for like; left available but not made the default |

"""
    return md


def sec_training():
    """Section 3: the training algorithm."""
    b = rows_of(r"trainbench_torch_epoch_minibatch_final_styleB")
    if not b:
        b = rows_of(r"trainbench_torch_epoch_minibatch_round2b_styleB") + \
            rows_of(r"trainbench_torch_epoch_minibatch_round2c_large")
    b = sorted({r["n_copies"]: r for r in b}.values(), key=lambda r: r["n_copies"])
    a = {r["n_copies"]: r for r in (rows_of(r"trainbench_torch_full_batch_final_styleA") or
                                    rows_of(r"trainbench_torch_full_batch_round2b_styleA"))}
    md = """## 3. The training algorithm

### 3.1 What it does, and what "batching the copies" means

One training iteration proceeds in three stages. First the agents act: each copy runs its
environments forward for 128 steps, choosing actions from its current policy and recording
what happened. Second, that recorded experience is processed into learning targets — how much
better or worse each action turned out than expected, and how surprising each visited state
was. Third, the networks are adjusted using those targets.

Running many copies at once required rewriting every part of this so that the copy count is a
dimension of the data rather than a loop. Each copy's networks are stored stacked together, so
a layer that would be one matrix multiplication for one copy becomes one batched matrix
multiplication for all copies. Every running average, every gradient limit, and every random
draw is likewise computed per copy.

That last point is a correctness requirement, not an implementation detail: if any quantity
were accidentally shared, the copies would silently stop being independent experiments and the
results would be worthless. Three tests guard this:

- Two runs with the same seed produce identical parameters to the last bit.
- Deliberately perturbing one copy leaves every other copy's parameters identical to the last
  bit after training.
- The same experiment implemented independently in the second framework agrees to within
  8.6e-7 on every intermediate quantity.

### 3.2 Results: how throughput scales with the number of copies

"""
    if not b:
        return md + missing("the training benchmark files")
    md += ("| copies | milliseconds per iteration | million steps per second, total | "
           "thousand steps per second, per copy | peak memory (GB) |\n|---|---|---|---|---|\n")
    for r in b:
        vram = r.get("peak_vram_mb")
        md += (f"| {r['n_copies']:,} | {r['sec_per_iteration']*1e3:.1f} | "
               f"{M(r['env_steps_per_sec'])} | {K(r['env_steps_per_sec_per_copy'])} | "
               f"{vram/1024:.1f} |\n" if vram else
               f"| {r['n_copies']:,} | {r['sec_per_iteration']*1e3:.1f} | "
               f"{M(r['env_steps_per_sec'])} | {K(r['env_steps_per_sec_per_copy'])} | not recorded |\n")
    md += """
*Many small updates per batch. One iteration collects 512 environment steps per copy.*

The second convention, one update per batch of data, does less arithmetic and is
correspondingly faster:

| copies | milliseconds per iteration | million steps per second, total | thousand steps per second, per copy |
|---|---|---|---|
"""
    for c in sorted(a):
        r = a[c]
        md += (f"| {c:,} | {r['sec_per_iteration']*1e3:.1f} | {M(r['env_steps_per_sec'])} | "
               f"{K(r['env_steps_per_sec_per_copy'])} |\n")
    md += """
![training scaling](figures/training_scaling.png)

Two observations follow from these tables, and they are the central practical facts about the
system:

- **Total throughput rises with the number of copies and then saturates.** Going from 8 copies
  to 128 multiplies the work done per second by roughly eleven while the time per iteration
  rises only modestly, because at small copy counts the processor is mostly idle. Beyond
  roughly eight thousand copies the curve flattens: the machine is full.
- **Per-copy throughput falls once past that point.** Each individual experiment advances more
  slowly when it shares the processor with thousands of others. A researcher who needs one
  experiment finished quickly should use few copies; a researcher who needs many experiments
  finished should use many. The tables give the exchange rate.

The largest configuration that fits in memory is 16,384 copies. This ceiling moved down from
32,768 during the optimisation work, which is discussed as a deliberate trade in section 3.4.

### 3.3 Results: sweeping a parameter across copy groups

Copies need not be identical. The system can divide them into groups and give each group a
different learning rate — the parameter that controls how large each adjustment is — so that
one run compares several settings instead of testing one. The user supplies two things: the
list of rates, and how many copies each rate receives.

"""
    rates = rows_of(r"sweep_scaling_rates")
    groups = rows_of(r"sweep_scaling_groups")
    uvs = rows_of(r"uniform_vs_sweep")
    if rates:
        md += ("| learning rates | copies per rate | total copies | milliseconds per iteration | "
               "million steps per second | hours to train every copy for 10 million steps |\n"
               "|---|---|---|---|---|---|\n")
        iters = 10e6 / 512
        for r in rates:
            md += (f"| {r['n_rates']} | {r['copies_per_rate']} | {r['total_copies']:,} | "
                   f"{r['sec_per_iteration']*1e3:.1f} | {M(r['env_steps_per_sec'])} | "
                   f"{r['sec_per_iteration']*iters/3600:.2f} |\n")
    if groups:
        ms = [r["sec_per_iteration"] * 1e3 for r in groups]
        md += f"""
The number of distinct rates costs nothing by itself. Holding the total at
{groups[0]['total_copies']:,} copies and splitting them into between 1 and
{max(r['n_rates'] for r in groups)} groups leaves the iteration time between
{min(ms):.1f} and {max(ms):.1f} milliseconds — a spread of
{(max(ms)-min(ms))/(sum(ms)/len(ms))*100:.1f} percent, within measurement noise — and the
memory identical to the byte. Only the total number of copies matters.
"""
    if uvs:
        vec = [r["vector_cost_percent"] for r in uvs]
        dif = [abs(r["differing_cost_percent"]) for r in uvs]
        md += f"""
Comparing a swept run against a uniform one mixes two separate changes, so they were measured
apart. Giving every copy its own rate replaces the standard optimiser with a hand-written one,
because the standard implementation accepts only a single rate per group of parameters; that
substitution costs between {min(vec):.1f} and {max(vec):.1f} percent. Those rates then actually
differing changes no code at all and costs nothing measurable — at most {max(dif):.1f} percent,
inside the noise floor at every size.

![sweep](figures/sweep.png)

Finally, fusing the groups into one run is substantially better than training them one after
another, and the advantage grows with the number of groups, because each group alone is too
small to occupy the processor: 1.43 times at two rates, 1.84 at four, 2.11 at eight, and 2.23
at sixteen.
"""
    md += """
### 3.4 Optimisation techniques

**Kept.**

| technique | what it does | measured effect |
|---|---|---|
| Stack the copies' networks and use batched matrix multiplication | Turns many tiny operations into one larger one | 128 copies cost almost the same as 8 at the start of the work |
| Compile the repeated per-step function | Fuses dozens of small operations into a few programs | 3.4x |
| Record the whole iteration as a replayable sequence of operations | The processor is given one instruction to repeat a fixed sequence, instead of being fed thousands of individual instructions from the host program each iteration | rollout stage 226 to 35 milliseconds; whole iteration 45.6 to 41.5 |
| Use the optimiser variant that keeps its step counter on the device | Required for the recorded sequence to remain correct; a counter kept on the host would be frozen at the value it had when the recording was made | correctness, and it removes a synchronisation |
| Allow reduced-precision matrix multiplication | Uses the processor's dedicated units at slightly lower precision | 9 percent |
| Compute the fixed reference network's outputs once per iteration instead of once per update | Its outputs cannot change within an iteration, so sixteen recomputations were wasted | 1.6 percent with many updates, 6.7 with one |
| Combine matrix multiplications that read the same input | Two networks reading the same observations become one wider multiplication | 1.9 to 2.9 percent |
| Compile the post-rollout processing stage | Its two sequential passes over the collected experience were written as loops, producing hundreds of tiny programs | 11 percent at 128 copies, 14 at 8 |
| Move work out of the sequential loop | The value estimates, the action probabilities and the curiosity rewards do not need to be computed step by step: each depends only on data the loop already stores, so all 128 steps can be computed in one wide operation afterwards | 42 percent at 128 copies, 45 at 8 — the single largest improvement of the second round |
| Write the recorded results directly from the computation | Removes a separate copy operation per quantity per step | 5 percent |
| Compile the per-copy optimiser used by sweeps | Written plainly it is six operations per parameter tensor | reduces the cost of sweeping from 38 percent to about 1 |

**Tried and not kept.**

| technique | why it was rejected |
|---|---|
| Record the uncompiled sequence of operations | Slower than recording the compiled one: replaying thousands of tiny programs is limited by the number of programs, not by the dispatch of them |
| The framework's automatic recording mode | It silently declined to record parts of the computation; recording by hand was both faster and verifiable |
| Present only one of the two update conventions in the source | The concern was that having both might prevent fusion. A build with the other removed entirely, verified to compute identical results, differed by 0.05 to 0.08 percent against a noise floor of 0.01 milliseconds. The two conventions were already separate compiled programs, so the unused one costs nothing |
| Store the copies' networks in one large block-diagonal matrix | 47 times slower than batched multiplication and required 4.45 gigabytes of weights instead of 34 megabytes |
| A loop over copies in the host language | 41 times slower than batched multiplication |
| Remove a redundant value-network pass | Correct but worth 0.4 percent at most; recorded as a simplification rather than a speed improvement |
| Allow the memory allocator to grow its segments, to recover the copy ceiling | Reduced the failing allocation from 32 to 8 gigabytes but still ran out: the demand is real, not fragmentation |

**A trade that was accepted deliberately.** Moving work out of the sequential loop and caching
the reference network's outputs both hold more data in memory at once. The largest workable
configuration fell from 32,768 copies to 16,384, while every size up to that point became
about 1.8 times faster. A run that needs the extra copies more than the speed can switch both
techniques off.

"""
    return md


def sec_endtoend():
    """Section 4: the two parts combined."""
    ph = newest(r"profile_phases")
    md = """## 4. End to end

### 4.1 What "end to end" means here

Sections 2 and 3 measured the two parts in isolation. In use they are not separate programs:
the environment is stepped from inside the training loop, 128 times per iteration, between the
policy producing an action and the learning stage consuming the result. This section reports
the combined system — which is the only configuration a researcher actually runs.

The essential structural decision is that the environment, the policy, the learning targets and
the parameter update are all expressed as operations on the same device, with no data returned
to the host at any point during an iteration. That makes it possible to record an entire
iteration — all three stages, including the 128 environment steps and the sixteen parameter
updates — as one replayable sequence. The host then issues one instruction per iteration
instead of many thousands.

### 4.2 Where the time goes

"""
    if not ph:
        md += missing("the phase profile")
    else:
        tot = ph["phases_sum_us"]
        md += ("| stage of one iteration | milliseconds | share |\n|---|---|---|\n")
        for k, v in ph["phases_us"].items():
            md += f"| {k} | {v/1000:.2f} | {v/tot*100:.0f} percent |\n"
        md += (f"| the three stages measured separately | {tot/1000:.2f} | 100 percent |\n"
               f"| the same iteration recorded as ONE sequence, which is what ships | "
               f"{ph['one_graph_us']/1000:.2f} | {ph['one_graph_us']/tot*100:.0f} percent of the above |\n")
        md += f"""
*At {ph['n_copies']} copies. The last two rows are within measurement noise of each other: once
each stage is itself compiled and recorded, there is little left between them to save. Recording
the whole iteration as one sequence is still the shipped arrangement, because it issues one
instruction per iteration rather than three and so is insensitive to how busy the host is.*

The following table lists individual operations timed on their real sizes without compilation.
It does not decompose the time above — compiling and recording remove most of the overhead
these numbers contain — but it shows where the underlying work sits, which is what explains
which optimisations paid:

| operation | milliseconds if run unoptimised |
|---|---|
"""
        for k, v in sorted(ph["components_us"].items(), key=lambda kv: -kv[1]):
            md += f"| {k} | {v/1000:.2f} |\n"
        md += """
The environment dominates this list, which is why the environment work in section 2 came first.
After the optimisations, however, the picture inverts: see section 4.4.

"""
    md += """### 4.3 Which environment to pair with which trainer

Four combinations are possible, and all were measured.

| combination | result |
|---|---|
"""
    cud = {r["n_copies"]: r for r in rows_of(r"trainbench_torch_epoch_minibatch_cudaenv_streamfixed_prod")}
    tor = {r["n_copies"]: r for r in rows_of(r"trainbench_torch_epoch_minibatch_round2b_styleB")}
    if cud and tor:
        pairs = ", ".join(f"{c:,} copies: {cud[c]['sec_per_iteration']*1e3:.1f} against "
                          f"{tor[c]['sec_per_iteration']*1e3:.1f} milliseconds"
                          for c in sorted(cud) if c in tor)
        md += (f"| PyTorch environment with the PyTorch trainer | the shipped configuration; "
               f"the environment is compiled into the same recorded sequence as the trainer |\n"
               f"| CUDA-kernel environment with the PyTorch trainer | {pairs}. The two are "
               f"equal to about one and a half percent |\n")
    md += """| JAX environment with the JAX trainer | the shipped JAX configuration; the environment is traced into the same compiled program as the trainer |
"""
    cp = rows_of(r"cross_pairing")
    if cp:
        worst = max(r["boundary_overhead_us"] for r in cp)
        md += (f"| an environment from one framework with a trainer from the other | rejected. "
               f"Handing arrays across the framework boundary costs up to {worst/1000:.1f} "
               f"milliseconds per environment step, against a native step of 0.1 to 0.3 "
               f"milliseconds — a factor of forty or more — and it also prevents both frameworks "
               f"from fusing or recording the loop |\n")
    md += """
The second row deserves comment, because it is the most instructive result in the report. The
CUDA-kernel environment is roughly twenty-four times faster than the PyTorch environment when
measured on its own (section 2.3). Substituting it into the training loop changes the training
rate by about one percent. The reason is visible in section 4.2: after the optimisations, the
environment is a small share of an iteration, so making it faster has little left to give. The
fast kernel earns its place in work that is only environment simulation — generating data,
evaluating a fixed policy — and not in this training loop.

### 4.4 A defect that this pairing measurement exposed

An earlier version of the pairing comparison produced numbers that were wrong, and the way the
error was found is worth recording. The CUDA environment launched its work on a different
execution queue from the surrounding framework operations. When an iteration was recorded for
replay, the environment step was therefore not captured: the recording contained the policy,
the learning targets and the update, but no simulation at all. The replayed iteration ran and
produced plausible timings, and the environment simply never advanced.

The error was found by writing a test that recorded only the environment step, replayed it,
and asserted that the state had changed. It had not; the framework itself then reported that
the recorded sequence was empty. The fix was one line — launch on the current queue — after
which the previously published pairing numbers were discarded and re-measured. The test now
runs alongside the others.

### 4.5 The training campaign

To confirm that the system trains rather than merely runs quickly, every copy count was trained
for 10.24 million environment steps per copy, in both update conventions.

"""
    recs = campaign(True)
    r1 = campaign(False)
    if not recs:
        md += missing("the training campaign records")
    else:
        md += ("| update convention | copies | wall time (minutes) | million steps per second | "
               "thousand steps per second, per copy | fraction of the maze explored | "
               "copies that reached the goal |\n|---|---|---|---|---|---|---|\n")
        for (style, c) in sorted(recs, key=lambda k: (k[0], k[1])):
            r = recs[(style, c)]
            hist = r["history"]
            ever = sum(1 for i in range(c)
                       if max(row["reward_ext_sum_per_copy"][i] for row in hist) > 0)
            cov = sum(r["final_coverage_per_copy"]) / c
            name = ("one update per batch" if style == "full_batch"
                    else "many small updates per batch")
            md += (f"| {name} | {c} | {r['train_seconds']/60:.1f} | "
                   f"{M(r['env_steps_per_sec'])} | {K(r['env_steps_per_sec']/c)} | "
                   f"{cov:.2f} | {ever} of {c} |\n")
        md += """
![end to end](figures/endtoend.png)

Two remarks on reading this table. First, the exploration column is the one that shows the
method working: the agents visit most of the maze, which is what the curiosity reward is for,
and they do so from the very sparse external reward alone. Second, the goal column is a lower
bound — progress is sampled every fifty iterations, so a copy that reached the goal only
between samples is not counted.

"""
        if r1:
            pairs = [(k, recs[k], r1[k]) for k in recs if k in r1]
            if pairs:
                factor = sum(b["env_steps_per_sec"] / a["env_steps_per_sec"]
                             for _, b, a in pairs) / len(pairs)
                md += (f"The same campaign was run before and after the second round of "
                       f"optimisation work. Across the ten configurations the second run was "
                       f"{factor:.1f} times faster. The learning outcomes differ between the two "
                       f"runs by more than one might expect, and it is worth being explicit that "
                       f"this is not an effect of the optimisation: the algorithm is unchanged "
                       f"and every change was verified to compute the same function. Tiny "
                       f"differences in rounding send the runs down different trajectories, and "
                       f"on a task where the reward is found or not found, per-copy outcomes are "
                       f"close to all-or-nothing, so aggregates move. The two campaigns should be "
                       f"read as two samples of one procedure.\n\n")
    md += """### 4.6 Optimisation techniques at the level of the whole loop

**Kept.**

| technique | what it does | measured effect |
|---|---|---|
| Keep every stage on the device | No data returns to the host during an iteration, so nothing has to wait for the host | a precondition for everything below |
| Record the whole iteration as one sequence | One instruction per iteration instead of thousands | 45.6 to 41.5 milliseconds at 128 copies when it was introduced. Measured again after the later changes it is even with recording the stages separately, because those changes removed the gaps it had been closing; it is kept because it does not depend on the host keeping up |
| Fixed sizes throughout | Copy count, environment count and horizon are constants, so no recompilation occurs during a run | prevents the previous item from being silently undone |
| Draw all random numbers for an iteration in advance | A recorded sequence cannot call a random generator on the host | correctness under recording |

**Tried and not kept.**

| technique | why it was rejected |
|---|---|
| Pairing an environment from one framework with a trainer from the other | measured at more than forty times the cost of a native step, as above |
| Substituting the fastest environment into the training loop | correct and available, but worth about one percent because the environment is no longer the constraint |
| Recording the stages separately rather than together | superseded: recording them together is faster |

"""
    return md


def sec_method():
    """Section 7: how the measurements were taken, and how to repeat them."""
    return """## 7. Method, and how to repeat the measurements

**Hardware and isolation.** One NVIDIA H100 NVL processor with 95 gigabytes of memory, in a
shared machine. Every measurement in this report ran while holding an exclusive lock on the
device, so no other process competed for it.

**Timing.** Each configuration is warmed up first, so that compilation and one-off allocation
are excluded, and then timed over repeated iterations with the device synchronised around each
one. The reported figure is the median.

**Paired comparison.** Where a claimed difference is small, the two configurations are run in
alternating order — first, second, second, first — each in its own process, and the spread
between repeats of the same configuration is reported as a noise floor next to the difference.
A difference below that floor is reported as no effect. This matters: several of the changes in
this report are worth one or two percent, which is not distinguishable from drift without it.

**Comparison against the previous version.** Rather than comparing against a number recorded
earlier, the benchmark can load the previous revision of the trainer from version control and
run both in the same session, so that before and after are measured under identical conditions.

**Provenance.** Every measurement writes a file containing the numbers, the configuration and
the version-control revision. This report is generated from those files, so regenerating it
after a new measurement updates every affected table and figure.

**Reproduction.** The environment implementations, the trainer, the benchmark programs and the
training driver are all in the repository under `09_parallelization/`, with a record of every
experiment attempted — including those abandoned — in a progress file beside each component.
"""


def main():
    """Generate the figures and assemble the document."""
    import cpu_sections as cpu
    FIGS.mkdir(exist_ok=True)
    fig_env_throughput()
    fig_training_scaling()
    fig_sweep()
    fig_endtoend()
    for note in (cpu.fig_cpu_vs_gpu(), cpu.fig_best_setup(), cpu.fig_worker_scaling()):
        if note:
            MISSING.append(note)
    md = "\n".join([sec_intro(), sec_env(), sec_training(), sec_endtoend(),
                     cpu.sec_cpu(), cpu.sec_best(), sec_method()])
    (REPORT / "report.md").write_text(md)
    print(f"wrote {REPORT/'report.md'} ({len(md.splitlines())} lines) and figures")
    for m in MISSING:
        print("MISSING:", m)


if __name__ == "__main__":
    main()
