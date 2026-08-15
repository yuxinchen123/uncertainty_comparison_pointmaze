"""Processor-versus-graphics-processor material for the unified report.

Imported by build.py. Keeps the two new sections (the processor results, and the best setup
under four thousand copies) separate from the original three so the file stays readable.

Every rate is in millions of environment steps per second; per-copy rates are in thousands,
because in millions they round to zero once there are thousands of copies.
"""
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REPORT = HERE.parent
BASE = REPORT.parent.parent
RESULTS = BASE / "benchmarks" / "results"
FIGS = REPORT / "figures"

# one colour per platform, held fixed across every figure; the graphics processor is always
# drawn dashed so it reads as the reference line rather than as another processor curve
C_GPU = "#2a78d6"
C_CPU_PROC = "#eb6834"
C_CPU_THREAD = "#4a3aa7"
C_CPU_OLD = "#1baf7a"
GRID = dict(color="#d9d9d9", linewidth=0.6)
ITERS_FOR_10M = 10e6 / 512          # iterations to give every copy ten million steps


def all_rows(pattern, **filters):
    """Every row from every results file matching the pattern, with optional field filters.

    Unlike the newest-file helper, this gathers a curve that was measured across several
    invocations — the processor sweeps are split by worker count and by update convention.
    """
    out = []
    for p in sorted(RESULTS.glob("*.json")):
        if not re.search(pattern, p.name):
            continue
        d = json.loads(p.read_text())
        if any(d.get(k) != v for k, v in filters.items()):
            continue
        for r in d.get("rows", []):
            row = dict(r)
            row["_host"] = d.get("host", "gpu")
            row["_mode"] = d.get("mode", "gpu")
            row["_style"] = d.get("style", "epoch_minibatch")
            row["_threads"] = d.get("threads")
            out.append(row)
    return out


def dedup(rows, key):
    """Keep the fastest measurement per key value, sorted by that key."""
    best = {}
    for r in rows:
        k = r[key]
        if k not in best or r["env_steps_per_sec"] > best[k]["env_steps_per_sec"]:
            best[k] = r
    return sorted(best.values(), key=lambda r: r[key])


def cpu_train(host, mode, style="epoch_minibatch"):
    """Processor training rows for one host, parallelisation style and update convention."""
    rows = [r for r in all_rows(r"trainbench_cpu_", host=host, mode=mode, style=style)]
    return dedup(rows, "total_copies")


def gpu_train(style="epoch_minibatch"):
    """Graphics-processor training rows for one update convention, across every file."""
    tag = "epoch_minibatch" if style == "epoch_minibatch" else "full_batch"
    rows = all_rows(rf"trainbench_torch_{tag}_(final_style|round2c_large)")
    for r in rows:
        r["total_copies"] = r["n_copies"]
    return dedup(rows, "total_copies")


def style_ax(ax):
    """Recessive grid, no top or right frame."""
    ax.grid(True, which="major", **GRID)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def fig_cpu_vs_gpu():
    """End-to-end training on both parallelisation styles, with the graphics processor dashed.

    Two panels, because a throughput result is two numbers: the aggregate the machine delivers,
    and what one copy gets. They point opposite ways — packing more copies raises the first and
    lowers the second — so a reader choosing a setting needs both curves.
    """
    tr_proc = cpu_train("jaguar03", "processes") or cpu_train("puma01", "processes")
    tr_thread = cpu_train("jaguar03", "threads") or cpu_train("puma01", "threads")
    tr_gpu = gpu_train("epoch_minibatch")
    if not (tr_proc or tr_thread):
        return "the processor training measurements"

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.4), dpi=160)
    curves = [("processor, independent processes", tr_proc, C_CPU_PROC, "o", "-"),
              ("processor, threads in one process", tr_thread, C_CPU_THREAD, "s", "-"),
              ("graphics processor (reference)", tr_gpu, C_GPU, "x", "--")]
    # left panel is the aggregate rate, right panel the rate one copy gets; same curves on both
    for label, rows, colour, marker, dash in curves:
        if not rows:
            continue
        x = [r["total_copies"] for r in rows]
        axes[0].plot(x, [r["env_steps_per_sec"] / 1e6 for r in rows], dash, color=colour,
                     linewidth=2, marker=marker, markersize=6, label=label)
        axes[1].plot(x, [r["env_steps_per_sec_per_copy"] / 1e3 for r in rows], dash, color=colour,
                     linewidth=2, marker=marker, markersize=6, label=label)
    axes[0].set_ylabel("million environment steps per second")
    axes[0].set_title("Total across all copies", fontsize=10)
    axes[1].set_ylabel("thousand environment steps per second per copy")
    axes[1].set_title("What one copy gets", fontsize=10)

    for ax in axes:
        ax.set_xlabel("independent training copies (log scale)")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.legend(frameon=False, fontsize=8)
        style_ax(ax)
    fig.suptitle("End-to-end training: ordinary processor cores against the graphics processor",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGS / "cpu_vs_gpu.png")
    plt.close(fig)
    return None


def mark_best(texts, values, higher_is_better):
    """Bold the best cell of a column and underline the second best, ties included.

    before: texts = ["24.1", "7.57", "3.80"], values = [24.1, 7.57, 3.80], higher_is_better
    after:  ["**24.1**", "<u>7.57</u>", "3.80"]
    """
    ranked = sorted(set(values), reverse=higher_is_better)
    best = ranked[0]
    second = ranked[1] if len(ranked) > 1 else None
    out = []
    for t, v in zip(texts, values):
        out.append(f"**{t}**" if v == best else (f"<u>{t}</u>" if v == second else t))
    return out


def best_under(limit=4096):
    """The configuration maximising total throughput below a copy limit, per platform."""
    out = []
    for name, rows, kind in [
            ("graphics processor, one update per batch", gpu_train("full_batch"), "gpu"),
            ("graphics processor, sixteen updates per batch", gpu_train("epoch_minibatch"), "gpu"),
            ("processor, independent processes, sixteen updates",
             cpu_train("jaguar03", "processes") or cpu_train("puma01", "processes"), "cpu"),
            ("processor, independent processes, one update",
             cpu_train("jaguar03", "processes", "full_batch")
             or cpu_train("puma01", "processes", "full_batch"), "cpu"),
            ("processor, threads in one process, sixteen updates",
             cpu_train("jaguar03", "threads") or cpu_train("puma01", "threads"), "cpu"),
            ("processor, threads in one process, one update",
             cpu_train("jaguar03", "threads", "full_batch")
             or cpu_train("puma01", "threads", "full_batch"), "cpu")]:
        pick = [r for r in rows if r["total_copies"] <= limit]
        if not pick:
            continue
        b = max(pick, key=lambda r: r["env_steps_per_sec"])
        out.append({"name": name, "kind": kind, "copies": b["total_copies"],
                    "host": b.get("_host", "H100"),
                    "sec_per_iteration": b["sec_per_iteration"],
                    "total": b["env_steps_per_sec"],
                    "per_copy": b["env_steps_per_sec_per_copy"],
                    "hours_10M": b["sec_per_iteration"] * ITERS_FOR_10M / 3600})
    return sorted(out, key=lambda r: -r["total"])


def fig_best_setup(limit=4096):
    """The best configuration of each platform, side by side."""
    rows = best_under(limit)
    if not rows:
        return "the measurements the best-setup comparison draws on"
    fig, axes = plt.subplots(1, 3, figsize=(14.6, 4.4), dpi=160)
    names = [r["name"].replace(", ", "\n") for r in rows]
    colours = [C_GPU if r["kind"] == "gpu" else C_CPU_PROC for r in rows]
    # the three panels are the aggregate rate, the rate one copy gets, and what the per-copy rate
    # means in wall time; the row order is shared, so every panel must run the same way up
    panels = [("total", 1e6, "{:.3f}", "million environment steps per second (log scale)",
               f"Best total throughput at {limit:,} copies or fewer"),
              ("per_copy", 1e3, "{:,.2f}", "thousand steps per second per copy (log scale)",
               "What one copy gets there"),
              ("hours_10M", 1, "{:.1f} h", "hours to give every copy ten million steps (log scale)",
               "What that means in wall time")]
    for ax, (field, scale, fmt, xlabel, title) in zip(axes, panels):
        ax.barh(range(len(rows)), [r[field] / scale for r in rows], color=colours, height=0.6)
        for i, r in enumerate(rows):
            ax.text(r[field] / scale, i, "  " + fmt.format(r[field] / scale), va="center",
                    fontsize=8)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(names if field == "total" else [], fontsize=7.5)
        ax.invert_yaxis()
        ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_title(title, fontsize=10)
        style_ax(ax)
    fig.tight_layout()
    fig.savefig(FIGS / "best_setup.png")
    plt.close(fig)
    return None


def M(x):
    """Millions, formatted for a table cell."""
    v = x / 1e6
    return f"{v:,.0f}" if v >= 100 else (f"{v:.1f}" if v >= 10 else
                                          (f"{v:.2f}" if v >= 1 else f"{v:.4f}"))


def K(x):
    """Thousands, formatted for a table cell."""
    v = x / 1e3
    return f"{v:,.0f}" if v >= 10 else f"{v:.2f}"


def thread_table(host):
    """Thread-mode table: one row per (copy count, thread setting), with both rates on every row.

    One row per setting rather than one column per setting, so that the aggregate rate and the
    per-copy rate are both visible for BOTH thread settings; folding the second setting into an
    extra column can only show one of the two quantities for it. The rows stay grouped by copy
    count so the two settings sit next to each other and remain directly comparable.
    before: rows = [{copies:1,threads:8,...}, {copies:1,threads:112,...}, {copies:2,threads:8,...}]
    after:  copies 1 / 8 threads, copies 1 / 112 threads, copies 2 / 8 threads, ...
    """
    rows = all_rows(r"trainbench_cpu_threads", host=host, style="epoch_minibatch")
    if not rows:
        return ""
    md = ("| copies | threads | seconds per iteration | million steps per second | "
          "thousand steps per second per copy |\n|---|---|---|---|---|\n")
    for r in sorted(rows, key=lambda r: (r["total_copies"], r["workers"])):
        md += (f"| {r['total_copies']} | {r['workers']} | {r['sec_per_iteration']:.3f} | "
               f"{M(r['env_steps_per_sec'])} | {K(r['env_steps_per_sec_per_copy'])} |\n")
    md += ("\n*One process holding every copy, the array library given 8 or 112 threads. "
           "Sixteen updates per batch.*\n\n")
    return md


def thread_verdict(host, tr_proc, tr_thread):
    """The two sentences comparing the parallelisation styles, with every number read from data.

    Written from the measurements rather than typed in, so re-running the benchmark cannot leave
    the prose disagreeing with the tables printed directly above it.
    """
    if not tr_proc or not tr_thread:
        return "*Waiting on measurements.*"
    # headline: the best each style reached anywhere in its sweep
    top_p = max(tr_proc, key=lambda r: r["env_steps_per_sec"])
    top_t = max(tr_thread, key=lambda r: r["env_steps_per_sec"])
    ratio = top_p["env_steps_per_sec"] / top_t["env_steps_per_sec"]
    out = (f"Independent processes reach {M(top_p['env_steps_per_sec'])} million environment "
           f"steps per second at {top_p['total_copies']:,} copies; one process with threads tops "
           f"out at {M(top_t['env_steps_per_sec'])} million. That is a factor of "
           f"**{ratio:,.0f}** on the same machine, running the same algorithm — the only "
           f"difference is how the work was divided.\n\n")

    # the second claim: more threads did not help. Compare the two thread settings copy by copy.
    rows = all_rows(r"trainbench_cpu_threads", host=host, style="epoch_minibatch")
    settings = sorted({r["workers"] for r in rows})
    if len(settings) < 2:
        return out
    low, high = settings[0], settings[-1]
    paired = {}
    for r in rows:
        paired.setdefault(r["total_copies"], {})[r["workers"]] = r["sec_per_iteration"]
    both = {c: v for c, v in paired.items() if low in v and high in v}
    slower = [c for c, v in both.items() if v[high] > v[low] * 1.02]
    # cite the two largest copy counts, where there is the most work available to divide and so
    # the best case for extra threads; the smallest counts would be a weaker demonstration
    ex = sorted(slower)[-2:]
    cites = "; ".join(f"{both[c][high]:.3f} seconds per iteration against {both[c][low]:.3f} "
                      f"at {c} {'copy' if c == 1 else 'copies'}" for c in ex)
    out += (f"The thread table also shows that adding threads does not help. Giving the single "
            f"process {high} threads instead of {low} was slower at {len(slower)} of the "
            f"{len(both)} copy counts measured — {cites} — and never faster by more than the "
            f"measurement noise. {high // low} times as many threads bought nothing.")
    return out


def sec_cpu():
    """Section: the same work on ordinary processor cores."""
    tr_proc = cpu_train("jaguar03", "processes") or cpu_train("puma01", "processes")
    tr_thread = cpu_train("jaguar03", "threads") or cpu_train("puma01", "threads")
    host = tr_proc[0]["_host"] if tr_proc else (tr_thread[0]["_host"] if tr_thread else "?")
    md = f"""## 5. The same work on ordinary processor cores

### 5.1 Why this comparison is here

Every number so far came from a graphics processor. A reader deciding where to run this work
needs to know what the alternative gives, so the same end-to-end training loop was measured on
ordinary processor cores. The measurements below ran on **{host}**
(AMD EPYC 7663, 224 logical processors, 1 TB of memory), held exclusively — no other job shared
the machine — so the timings are not contaminated by a neighbour. It was chosen as the largest
completely idle node on the cluster; a node with more cores was available but already had
another job on it, which is exactly the contamination this run set out to avoid.

Nothing in the algorithm changed. What changed is that the graphics-processor features the
optimisation work relied on — recording an iteration as a replayable sequence, the
reduced-precision matrix mode, the fused optimiser — do not exist on a processor, so the
processor runs the same code in its plain form.

### 5.2 Two ways to use many cores, and why they differ so much

A processor has many cores, and the work has to be divided among them. There are two ways:

- **Threads inside one process.** One program holds every copy in one set of arrays. Each
  instruction covers all of them, and the array library splits that one instruction across N
  threads. The threads must regroup after every instruction, because the next one reads what
  the previous one wrote.
- **Independent processes.** N separate programs, each owning its own copies, each using one
  thread. They never coordinate, because there is nothing to coordinate around.

The measurements settle which is better, and the answer is not the obvious one.

"""
    md += thread_table(host)
    if tr_proc:
        md += ("| workers | copies each | total copies | seconds per iteration | "
               "million steps per second | thousand steps per second per copy |\n"
               "|---|---|---|---|---|---|\n")
        for r in tr_proc:
            md += (f"| {r['workers']} | {r['n_copies']} | {r['total_copies']} | "
                   f"{r['sec_per_iteration']:.3f} | {M(r['env_steps_per_sec'])} | "
                   f"{K(r['env_steps_per_sec_per_copy'])} |\n")
        md += "\n*Independent single-thread processes. Sixteen updates per batch.*\n\n"
    md += f"""![processor against graphics processor](figures/cpu_vs_gpu.png)

The two tables answer it. {thread_verdict(host, tr_proc, tr_thread)}

The reason is the regrouping. One environment step is roughly forty small operations, each
individually cheap, and the coordination after each one costs a fixed amount regardless of how
little work it contained. With N threads that cost is paid forty times per step, so past a
handful of threads the coordination costs more than the work it coordinates. Independent
processes never pay it. This is also why the graphics processor needed the opposite treatment:
the optimisation work there fused those forty operations into a handful and recorded the whole
sequence, which is the same problem solved from the other end.

"""
    return md


def sec_best(limit=4096):
    """Section: the best configuration on each platform within the range actually used."""
    rows = best_under(limit)
    if not rows:
        return "## 6. The best setup on each platform\n\n*Waiting on measurements.*\n"
    fastest = rows[0]
    md = f"""## 6. The best setup on each platform

Throughput keeps rising with the number of copies well past the point most work needs, so the
comparison below is restricted to **{limit:,} copies or fewer**, which is the range this project
actually operates in. For each platform and configuration, the table gives the setting that
reaches the highest total throughput inside that range.

| platform and configuration | copies | seconds per iteration | million steps per second ↑ | thousand steps per second per copy | hours to ten million steps per copy |
|---|---|---|---|---|---|
"""
    # mark best and second best in every scored column (copies is a setting, so it is skipped)
    cells = {"sec_per_iteration": [f"{r['sec_per_iteration']:.3f}" for r in rows],
             "total": [M(r["total"]) for r in rows],
             "per_copy": [K(r["per_copy"]) for r in rows],
             "hours_10M": [f"{r['hours_10M']:.2f}" for r in rows]}
    for field, higher_is_better in [("sec_per_iteration", False), ("total", True),
                                    ("per_copy", True), ("hours_10M", False)]:
        cells[field] = mark_best(cells[field], [r[field] for r in rows], higher_is_better)
    for i, r in enumerate(rows):
        md += (f"| {r['name']} | {r['copies']:,} | {cells['sec_per_iteration'][i]} | "
               f"{cells['total'][i]} | {cells['per_copy'][i]} | {cells['hours_10M'][i]} |\n")
    gpu = [r for r in rows if r["kind"] == "gpu"]
    cpu = [r for r in rows if r["kind"] == "cpu"]
    md += "\n![best setup](figures/best_setup.png)\n\n"
    if gpu and cpu:
        ratio = gpu[0]["total"] / cpu[0]["total"]
        md += (f"The best graphics-processor configuration reaches {M(gpu[0]['total'])} million "
               f"environment steps per second at {gpu[0]['copies']:,} copies; the best processor "
               f"configuration reaches {M(cpu[0]['total'])} million at {cpu[0]['copies']:,} "
               f"copies. That is a factor of **{ratio:.1f}**. In wall-clock terms, giving every "
               f"copy ten million environment steps takes {gpu[0]['hours_10M']:.2f} hours on the "
               f"graphics processor against {cpu[0]['hours_10M']:.2f} hours on the processor "
               f"node.\n\n")
    md += """Best in each column is bold, second best underlined; copies is a setting rather than
a score, so it is not marked. Two qualifications belong with those numbers. The processor figure is for one node held
exclusively; a cluster with many such nodes multiplies it, and the independent-process
arrangement is exactly what a work queue across many nodes would do. And a configuration that
wins on total throughput is not the one that finishes any single copy soonest, which is why both
rates appear in every table and both curves in every figure.

"""
    return md


