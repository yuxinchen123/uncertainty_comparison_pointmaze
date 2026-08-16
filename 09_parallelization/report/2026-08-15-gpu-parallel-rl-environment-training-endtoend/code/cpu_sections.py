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
            # the methodology fields, absent from every file written before they were added
            row["_iters"] = d.get("iters")
            row["_aggregate"] = d.get("aggregate")
            row["_file"] = p.name
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


# a measurement that timed less than this many seconds of work is an opening burst: this
# processor runs above its sustained clock for the first seconds of a load, and a benchmark that
# adds up hundreds of processes' own rates over so short a window describes a load that never
# existed on the machine at one time. Both effects grow with the worker count.
SUSTAINED_SECONDS = 45.0
DEFAULT_ITERS = 5      # what every file written before the iteration count was recorded used
# two sets of files were measured at 150 iterations before the benchmark recorded its iteration
# count, and their names are the only record of it: the `sustained_` and `confirm_` runs of the
# node-comparison job (its slurm scripts pass --iters 150). Named here so those rows are read as
# what they are rather than as five-iteration bursts.
LONG_RUN_TAGS = ("sustained", "confirm")
LONG_RUN_ITERS = 150


def declared_iters(row):
    """The iteration count a row was measured with, from the file or from the run that wrote it."""
    if row.get("_iters") is not None:
        return row["_iters"]
    if any(tag in row.get("_file", "") for tag in LONG_RUN_TAGS):
        return LONG_RUN_ITERS
    return DEFAULT_ITERS


def timed_seconds(row):
    """How many seconds of work a row's measurement actually timed.

    Three cases, one rule. A row from the barrier-synchronised sweep carries the length of its
    own window. A row from the plain benchmark carries its iteration count, so the length is that
    count times the seconds an iteration took. A row from a file written before the iteration
    count was recorded gets the benchmark's default of five.
    before: {"window_seconds": 92.4} / {"_iters": 150, "sec_per_iteration": 0.443} / {}
    after:  92.4 / 66.5 / 5 x its own seconds per iteration
    """
    if row.get("window_seconds") is not None:
        return row["window_seconds"]
    return declared_iters(row) * row["sec_per_iteration"]


def is_sustained(row):
    """Whether a row is a settled rate rather than the opening seconds of a load."""
    return timed_seconds(row) >= SUSTAINED_SECONDS


def method_rank(row):
    """How much a row's methodology is to be trusted, highest first.

    Sustained beats burst, because a burst reading is inflated by tens of percent at large worker
    counts. Within that, a rate measured over one wall-clock window shared by every worker beats
    a sum of each worker's own rate, which assumes an overlap it does not check.
    """
    return (is_sustained(row), row.get("window_seconds") is not None)


def median_row(rows):
    """The middle row of a set of repeats by total throughput, with the spread recorded.

    A median rather than the fastest: repeats of the same configuration differ by a fraction of a
    percent when the methodology is the same, and taking the best of them would bias every number
    upward by the size of the run-to-run noise.
    before: three repeats at 2.10, 2.12 and 2.11 million steps per second
    after:  the 2.11 row, carrying repeats 3 and a spread of 0.010
    """
    # the lower of the two middles when the count is even, so a pair of repeats reports the
    # slower of the two rather than the faster: taking the upper middle would make every
    # twice-measured setting the best of two, which is the bias a median is meant to avoid
    ordered = sorted(rows, key=lambda r: r["env_steps_per_sec"])
    chosen = dict(ordered[(len(ordered) - 1) // 2])
    lo, hi = ordered[0]["env_steps_per_sec"], ordered[-1]["env_steps_per_sec"]
    chosen["_repeats"] = len(ordered)
    chosen["_spread"] = hi / lo - 1.0 if lo > 0 else 0.0
    chosen["_sustained"] = is_sustained(chosen)
    chosen["_timed_seconds"] = timed_seconds(chosen)
    return chosen


def choose_per_setting(rows, keyfn):
    """One row per setting: the best methodology measured for it, and the median of its repeats.

    before: for (224 workers, 16 copies) there are two five-iteration files and three
            ninety-second ones
    after:  one row, the median of the three ninety-second ones, marked sustained
    """
    groups = {}
    for r in rows:
        groups.setdefault(keyfn(r), []).append(r)
    out = []
    for key, members in groups.items():
        best = max(method_rank(m) for m in members)
        out.append(median_row([m for m in members if method_rank(m) == best]))
    return out


def cpu_train(host, mode, style="epoch_minibatch"):
    """Processor training rows for one host, parallelisation style and update convention.

    One row per setting, where a setting is a worker count with a copy count — not a total copy
    count, since 112 workers of 32 copies and 224 workers of 16 are different settings that
    happen to run the same number of copies.
    """
    rows = [r for r in all_rows(r"trainbench_cpu_", host=host, mode=mode, style=style)]
    chosen = choose_per_setting(rows, lambda r: (r["workers"], r["n_copies"]))
    return sorted(chosen, key=lambda r: (r["total_copies"], r["workers"]))


def gpu_train(style="epoch_minibatch"):
    """Graphics-processor training rows for one update convention, across every file."""
    tag = "epoch_minibatch" if style == "epoch_minibatch" else "full_batch"
    rows = all_rows(rf"trainbench_torch_{tag}_(final_style|round2c_large)")
    for r in rows:
        r["total_copies"] = r["n_copies"]
    return dedup(rows, "total_copies")


def best_per_total(rows):
    """The best setting for each total copy count, for a figure whose axis is the copy count.

    Two settings can run the same number of copies — 112 workers of 32 copies and 224 workers of
    16 — and a curve drawn against copies would show both as one zigzag. What the axis asks is
    what the machine does with that many copies, so the better of the two answers it.
    before: rows at (112, 32) reaching 1.9 and (224, 16) reaching 2.1 million steps per second
    after:  one point at 3,584 copies, the 2.1 one
    """
    best = {}
    for r in rows:
        k = r["total_copies"]
        if k not in best or r["env_steps_per_sec"] > best[k]["env_steps_per_sec"]:
            best[k] = r
    return sorted(best.values(), key=lambda r: r["total_copies"])


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
    tr_proc_a = (cpu_train("jaguar03", "processes", "full_batch")
                 or cpu_train("puma01", "processes", "full_batch"))
    tr_thread = cpu_train("jaguar03", "threads") or cpu_train("puma01", "threads")
    tr_gpu = gpu_train("epoch_minibatch")
    if not (tr_proc or tr_thread):
        return "the processor training measurements"

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.4), dpi=160)
    # the axis is the copy count, so where two worker counts run the same number of copies the
    # better of them is the point drawn
    curves = [("processor, independent processes, sixteen updates", best_per_total(tr_proc),
               C_CPU_PROC, "o", "-"),
              ("processor, independent processes, one update", best_per_total(tr_proc_a),
               C_CPU_OLD, "^", "-"),
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
    """One row per named configuration: its best setting under a copy limit, or a pinned setting.

    Selection is by highest total throughput among the settings at or below the limit. A
    configuration may instead name one exact setting through `pin`, which is how the
    one-copy-per-worker processor row gets in: it is in the table for the rate one copy gets, not
    for its total, so selecting it by maximum total would pick a different setting entirely.
    """
    out = []
    for name, rows, kind, pin in [
            ("graphics processor, one update per batch", gpu_train("full_batch"), "gpu", None),
            ("graphics processor, sixteen updates per batch", gpu_train("epoch_minibatch"),
             "gpu", None),
            ("processor, independent processes, sixteen updates",
             cpu_train("jaguar03", "processes") or cpu_train("puma01", "processes"), "cpu", None),
            ("processor, independent processes, one update",
             cpu_train("jaguar03", "processes", "full_batch")
             or cpu_train("puma01", "processes", "full_batch"), "cpu", None),
            ("processor, independent processes, one update, one copy per worker",
             cpu_train("jaguar03", "processes", "full_batch"), "cpu", {"workers": 224,
                                                                      "n_copies": 1}),
            ("processor, threads in one process, sixteen updates",
             cpu_train("jaguar03", "threads") or cpu_train("puma01", "threads"), "cpu", None),
            ("processor, threads in one process, one update",
             cpu_train("jaguar03", "threads", "full_batch")
             or cpu_train("puma01", "threads", "full_batch"), "cpu", None)]:
        # one measured row per configuration: the pinned ones by an exact match on the setting,
        # the rest by the highest total throughput among the settings inside the copy limit
        # before: rows = [{workers: 8, total_copies: 8, ...}, ..., {workers: 224, n_copies: 1,
        #         total_copies: 224, ...}] and pin = {"workers": 224, "n_copies": 1}
        # after:  b = the 224-worker one-copy-each row, where an unpinned entry would have taken
        #         whichever row has the largest env_steps_per_sec
        pick = [r for r in rows if r["total_copies"] <= limit]
        if not pick:
            continue
        if pin:
            b, = [r for r in pick if all(r[k] == v for k, v in pin.items())]
        else:
            b = max(pick, key=lambda r: r["env_steps_per_sec"])
        out.append({"name": name, "kind": kind, "pinned": pin is not None,
                    "copies": b["total_copies"], "host": b.get("_host", "H100"),
                    "sec_per_iteration": b["sec_per_iteration"],
                    "total": b["env_steps_per_sec"],
                    "per_copy": b["env_steps_per_sec_per_copy"],
                    "hours_1M": hours_per_million(b["env_steps_per_sec_per_copy"])})
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
               f"Total throughput at {limit:,} copies or fewer"),
              ("per_copy", 1e3, "{:,.2f}", "thousand steps per second per copy (log scale)",
               "What one copy gets there"),
              ("hours_1M", 1, "{:.3f} h", "hours per million steps per copy (log scale)",
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
    """Thousands, formatted for a table cell.

    Three decimals below a thousand steps per second, because the largest settings of the
    copies-per-worker sweep give a copy single-digit steps per second and two decimals would
    print every one of them as 0.00.
    before: 1,104 -> "1.10";  263 -> "0.263";  8 -> "0.008"
    """
    v = x / 1e3
    if v >= 10:
        return f"{v:,.0f}"
    return f"{v:.2f}" if v >= 1 else f"{v:.3f}"


def hours_per_million(per_copy_rate):
    """Hours for one copy to take a million environment steps, from its steps-per-second rate.

    The per-copy rate restated as time, because the question asked of a rate is nearly always how
    long a run will take. Always normalised to a million steps per copy, whatever step budget the
    run at hand uses, so the figure is comparable across every table in the report.
    before: per_copy_rate = 1574.34 environment steps per second for one copy
    after:  1e6 / (3600 x 1574.34) = 0.1764 hours
    """
    return 1e6 / (3600 * per_copy_rate)


def H(per_copy_rate):
    """Hours per million steps per copy, formatted for a table cell.

    Three significant figures rather than three decimal places: a graphics-processor row sits
    near 0.008 hours and a processor row near 0.7, so a fixed number of decimals either rounds
    the fast rows together (8 and 16 copies both printing 0.008) or prints noise on the slow ones.
    before: 37,000 steps per second per copy -> "0.00751";  1,574 -> "0.176"
    """
    return f"{hours_per_million(per_copy_rate):.3g}"


def thread_table(host):
    """Thread-mode table: one row per (copy count, thread setting), with both rates on every row.

    One row per setting rather than one column per setting, so that the aggregate rate and the
    per-copy rate are both visible for BOTH thread settings; folding the second setting into an
    extra column can only show one of the two quantities for it. The rows stay grouped by copy
    count so the two settings sit next to each other and remain directly comparable.
    before: rows = [{copies:1,threads:8,...}, {copies:1,threads:112,...}, {copies:2,threads:8,...}]
    after:  copies 1 / 8 threads, copies 1 / 112 threads, copies 2 / 8 threads, ...
    """
    rows = cpu_train(host, "threads", "epoch_minibatch")
    if not rows:
        return ""
    any_burst = any(not r.get("_sustained") for r in rows)
    md = ("| copies | threads | seconds per iteration | million steps per second | "
          "thousand steps per second per copy | hours per million steps per copy |"
          + (" timing |\n" if any_burst else "\n")
          + ("|---|---|---|---|---|---|---|\n" if any_burst else "|---|---|---|---|---|---|\n"))
    for r in sorted(rows, key=lambda r: (r["total_copies"], r["workers"])):
        md += (f"| {r['total_copies']} | {r['workers']} | {r['sec_per_iteration']:.3f} | "
               f"{M(r['env_steps_per_sec'])} | {K(r['env_steps_per_sec_per_copy'])} | "
               f"{H(r['env_steps_per_sec_per_copy'])} |"
               + ((" sustained |\n" if r.get("_sustained")
                   else f" burst, {r['_timed_seconds']:.0f}s |\n") if any_burst else "\n"))
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
    rows = cpu_train(host, "threads", "epoch_minibatch")
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


def plateau():
    """The sweep's side measurements, or None when they have not been taken yet.

    The ladder rungs themselves are ordinary result files and are read through `cpu_train`; this
    file carries only what is not a throughput setting anyone would run — one worker alone on the
    node, the memory system's own rate, and what the memory system had left under load.
    """
    hits = sorted(RESULTS.glob("*_cpu_copies_per_worker_plateau.json"))
    return json.loads(hits[-1].read_text()) if hits else None


def ladder(style, workers):
    """One copies-per-worker ladder: the swept rungs at one worker count and update convention.

    Only rows carrying a shared measurement window, that is only rows from the sweep, so the
    ladder is one methodology from end to end and its rungs are comparable with each other.
    """
    rows = [r for r in cpu_train("jaguar03", "processes", style)
            if r["workers"] == workers and r.get("window_seconds") is not None]
    return sorted(rows, key=lambda r: r["n_copies"])


def GB(mb):
    """Megabytes as a gigabyte figure for a table cell."""
    return f"{mb / 1024:.2f}"


def gain(rows):
    """Each rung's total throughput as a fraction of the rung below it.

    before: rows = totals [2.0, 3.0, 3.03] million steps per second
    after:  [None, 0.500, 0.010] — the first rung has nothing below it to gain over
    """
    out = [None]
    for prev, cur in zip(rows, rows[1:]):
        out.append(cur["env_steps_per_sec"] / prev["env_steps_per_sec"] - 1.0)
    return out


def plateau_rung(rows, threshold=0.02):
    """The rung past which total throughput no longer rises: the first gain under the threshold.

    Returns the rung at which the curve has flattened, that is the first one that bought less
    than the threshold over the rung below it. None means every rung measured still gained more
    than that, so the curve had not flattened where the sweep stopped.
    before: totals [2.0, 3.0, 3.03, 3.04] with a threshold of 0.02
    after:  the third rung, which gained 1% over the second
    """
    for row, g in zip(rows, gain(rows)):
        if g is not None and g < threshold:
            return row
    return None


def correction_rows(host, workers, copies):
    """The same setting read three ways, to separate the two things the old method got wrong.

    The three readings, in the order the corrections were applied:
      five iterations, sum of rates     what the earlier sweep reported
      ninety seconds, sum of rates      the same arithmetic on a settled load
      ninety seconds, common window     the work the machine did while every worker was running
    before: for (224 workers, 16 copies) there is a five-iteration file and a swept row
    after:  one entry per update convention, carrying all three numbers
    """
    out = []
    for style, label in [("full_batch", "one update per batch"),
                         ("epoch_minibatch", "sixteen updates per batch")]:
        rows = [r for r in all_rows(r"trainbench_cpu_", host=host, mode="processes", style=style)
                if r["workers"] == workers and r["n_copies"] == copies]
        burst = [r for r in rows if not is_sustained(r)]
        window = [r for r in rows if r.get("window_seconds") is not None]
        if not (burst and window):
            continue
        b, w = median_row(burst), median_row(window)
        out.append({"style": style, "label": label,
                    "burst": b["env_steps_per_sec"],
                    "burst_seconds": timed_seconds(b),
                    "long_sum": w["env_steps_per_sec_sum_of_worker_rates"],
                    "window": w["env_steps_per_sec"],
                    "window_seconds": w["window_seconds"],
                    "repeats": w["_repeats"]})
    return out


def correction_table(host="jaguar03", workers=224, copies=16):
    """The correction to the published processor numbers, as a table of the three readings."""
    rows = correction_rows(host, workers, copies)
    if not rows:
        return ""
    md = (f"| update convention | five iterations, rates added up | "
          f"ninety seconds, rates added up | ninety seconds, one shared window | "
          f"what the correction removes |\n|---|---|---|---|---|\n")
    for r in rows:
        md += (f"| {r['label']} | {M(r['burst'])} | {M(r['long_sum'])} | {M(r['window'])} | "
               f"{100 * (1 - r['window'] / r['burst']):.0f}% |\n")
    return md + (f"\n*The same setting — {workers} workers holding {copies} copies each, "
                 f"{workers * copies:,} copies — measured three ways on the same node in the same "
                 f"job. Millions of environment steps per second.*\n\n")


def repeat_table(host="jaguar03"):
    """Every setting measured more than once the same way, with each reading and the spread.

    before: two runs of 224 workers at 64 copies each, 1.77 and 1.76 million steps per second
    after:  one row naming both and the 0.8% between them
    """
    groups = {}
    for r in all_rows(r"trainbench_cpu_", host=host, mode="processes"):
        if r.get("window_seconds") is None:
            continue
        groups.setdefault((r["_style"], r["workers"], r["n_copies"]), []).append(r)
    repeated = {k: v for k, v in groups.items() if len(v) > 1}
    if not repeated:
        return ""
    md = ("| update convention | workers | copies per worker | readings, million steps per "
          "second | spread |\n|---|---|---|---|---|\n")
    for (style, workers, copies), members in sorted(repeated.items(),
                                                    key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        vals = sorted(m["env_steps_per_sec"] for m in members)
        label = "one update" if style == "full_batch" else "sixteen updates"
        md += (f"| {label} | {workers} | {copies} | "
               f"{', '.join(M(v) for v in vals)} | "
               f"{100 * (vals[-1] / vals[0] - 1):.1f}% |\n")
    return md + ("\n*Repeats of the same setting under the same method. The tables above report "
                 "the lower reading where a setting was measured twice.*\n\n")


def process_table(rows, caption):
    """The independent-process table: both rates, the memory a worker held, and how it was timed.

    The timing column appears only while some row is still a five-iteration measurement. Once
    every setting has been measured over a long window the column would say the same thing on
    every row, so it disappears on its own and the caption carries the fact instead.
    """
    any_burst = any(not r.get("_sustained") for r in rows)
    head = ("| workers | copies each | total copies | seconds per iteration | "
            "million steps per second | thousand steps per second per copy | "
            "hours per million steps per copy | peak memory per worker (GB) |")
    rule = "|---|---|---|---|---|---|---|---|"
    if any_burst:
        head += " timing |"
        rule += "---|"
    md = head + "\n" + rule + "\n"
    for r in rows:
        # a row measured before this sweep recorded no memory; "not recorded" is what the report
        # says elsewhere for a measurement that was not taken, and a back-filled guess would be
        # indistinguishable from a measured figure
        mem = (GB(r["peak_rss_mb_max_worker"]) if r.get("peak_rss_mb_max_worker")
               else "not recorded")
        cells = (f"| {r['workers']} | {r['n_copies']} | {r['total_copies']:,} | "
                 f"{r['sec_per_iteration']:.3f} | {M(r['env_steps_per_sec'])} | "
                 f"{K(r['env_steps_per_sec_per_copy'])} | "
                 f"{H(r['env_steps_per_sec_per_copy'])} | {mem} |")
        if any_burst:
            cells += (" sustained |" if r.get("_sustained")
                      else f" burst, {r['_timed_seconds']:.0f}s |")
        md += cells + "\n"
    note = (" Rows marked burst were timed for the seconds shown, which reads the opening seconds "
            "of the load rather than the rate a run gets; they are the measurements that have not "
            "been retaken yet."
            if any_burst else
            " Every row is timed over a window of about ninety seconds, with all workers "
            "synchronised so the rate is work the machine really did while carrying the full "
            "load.")
    return md + f"\n*{caption}{note}*\n\n"


def floor_lookup():
    """The matrix-work floor per setting, in seconds per iteration, or an empty table.

    before: the side file's rows, each naming a worker count, a copy count and a convention
    after:  {(224, 16, "full_batch"): 0.412, ...}
    """
    d = plateau()
    if not d:
        return {}
    return {(r["workers"], r["n_copies"], r["style"]): r["floor_seconds_median_worker"]
            for r in d.get("matmul_floor", [])}


def ladder_table(rows, caption, style):
    """One update convention's copies-per-worker ladder, with memory and distance from the floor.

    The last column is the share of the iteration that the matrix work's own best case accounts
    for: the same matrix multiplies timed on their own, with the same worker count on the same
    node, divided by the iteration. The graphics-processor side of this report is measured the
    same way, so the two platforms are judged by the same standard.
    """
    if not rows:
        return ""
    floors = floor_lookup()
    any_floor = any((r["workers"], r["n_copies"], style) in floors for r in rows)
    head = ("| copies per worker | total copies | seconds per iteration | "
            "million steps per second | thousand steps per second per copy | "
            "hours per million steps per copy | peak memory per worker (GB) | "
            "node memory in use (GB) |")
    rule = "|---|---|---|---|---|---|---|---|"
    if any_floor:
        head += " percent of the matrix-work floor |"
        rule += "---|"
    md = head + "\n" + rule + "\n"
    for r in rows:
        cells = (f"| {r['n_copies']} | {r['total_copies']:,} | {r['sec_per_iteration']:.3f} | "
                 f"{M(r['env_steps_per_sec'])} | {K(r['env_steps_per_sec_per_copy'])} | "
                 f"{H(r['env_steps_per_sec_per_copy'])} | {GB(r['peak_rss_mb_max_worker'])} | "
                 f"{r['node_peak_used_gb']:,.0f} |")
        if any_floor:
            f = floors.get((r["workers"], r["n_copies"], style))
            cells += (f" {100 * f / r['sec_per_iteration']:.1f}% |" if f
                      else " not measured |")
        md += cells + "\n"
    return md + f"\n*{caption}*\n\n"


def fig_cpu_plateau():
    """The copies-per-worker ladder: what the machine delivers, what one copy gets, what it costs.

    A figure of its own rather than more points on the earlier one, because every rung here was
    timed for about a minute and the earlier sweep's points were timed for two or three seconds.
    Drawing them as one curve would show a step at the join that is a change of measurement, not
    a change of the machine. The third panel is memory, since copies per worker is what drives it
    and the question of whether the curve stops because the node fills up is answered there.
    """
    curves = [("one update, 224 workers", ladder("full_batch", 224), C_CPU_OLD, "^", "-"),
              ("one update, 112 workers", ladder("full_batch", 112), C_CPU_OLD, "^", "--"),
              ("sixteen updates, 224 workers", ladder("epoch_minibatch", 224), C_CPU_PROC,
               "o", "-"),
              ("sixteen updates, 112 workers", ladder("epoch_minibatch", 112), C_CPU_PROC,
               "o", "--")]
    if not any(rows for _, rows, _, _, _ in curves):
        return "the copies-per-worker sweep"

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.4), dpi=160)
    for label, rows, colour, marker, dash in curves:
        if not rows:
            continue
        x = [r["total_copies"] for r in rows]
        axes[0].plot(x, [r["env_steps_per_sec"] / 1e6 for r in rows], dash, color=colour,
                     linewidth=2, marker=marker, markersize=6, label=label)
        axes[1].plot(x, [r["env_steps_per_sec_per_copy"] / 1e3 for r in rows], dash, color=colour,
                     linewidth=2, marker=marker, markersize=6, label=label)
        axes[2].plot(x, [r["peak_rss_mb_max_worker"] / 1024 for r in rows], dash, color=colour,
                     linewidth=2, marker=marker, markersize=6, label=label)
        # the rung where the curve flattened, marked so the reader sees where the answer is
        flat = plateau_rung(rows)
        if flat:
            axes[0].plot([flat["total_copies"]], [flat["env_steps_per_sec"] / 1e6], "o",
                         markersize=13, markerfacecolor="none", markeredgecolor=colour,
                         markeredgewidth=1.6)
    axes[0].set_ylabel("million environment steps per second")
    axes[0].set_title("Total across all copies", fontsize=10)
    axes[1].set_ylabel("thousand environment steps per second per copy")
    axes[1].set_title("What one copy gets", fontsize=10)
    axes[2].set_ylabel("gigabytes of memory per worker")
    axes[2].set_title("What one worker holds", fontsize=10)
    for ax in axes:
        ax.set_xlabel("total copies on the node, 224 workers throughout (log scale)")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.legend(frameon=False, fontsize=8)
        style_ax(ax)
    fig.suptitle("Copies per worker on jaguar03: where total throughput stops rising", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGS / "cpu_plateau.png")
    plt.close(fig)
    return None


def plateau_numbers():
    """Every figure the copies-per-worker passage quotes, per convention and worker count."""
    out = {}
    for style in ("full_batch", "epoch_minibatch"):
        for workers in (112, 224):
            rows = ladder(style, workers)
            if not rows:
                continue
            out[(style, workers)] = {
                "rows": rows, "top": max(rows, key=lambda r: r["env_steps_per_sec"]),
                "flat": plateau_rung(rows), "gains": gain(rows), "largest": rows[-1]}
    return out


def by_copies(rows):
    """The rows of one sweep indexed by the copies each worker held."""
    return {r["n_copies"]: r for r in rows}


def contention_table(d):
    """The same copy count run alone, one worker per core, and two workers per core.

    A worker alone on the node has the whole memory system to itself, so its seconds per
    iteration is the cost with no competition at all. The gap between that and the same worker
    inside a full machine is what competition costs, and the gap between one worker per core and
    two is what sharing a core costs.
    """
    alone = by_copies(d.get("alone_full_batch", []))
    one = by_copies(ladder("full_batch", 112))
    full = by_copies(ladder("full_batch", 224))
    shared = sorted(set(alone) & set(one) & set(full))
    if not shared:
        return ""
    md = ("| copies per worker | seconds per iteration, one worker alone | "
          "seconds per iteration, 112 workers | seconds per iteration, 224 workers | "
          "slowdown at 112 workers | slowdown at 224 workers | "
          "million steps per second, 112 workers | million steps per second, 224 workers |\n"
          "|---|---|---|---|---|---|---|---|\n")
    for c in shared:
        a, o, f = alone[c], one[c], full[c]
        md += (f"| {c} | {a['sec_per_iteration']:.3f} | {o['sec_per_iteration']:.3f} | "
               f"{f['sec_per_iteration']:.3f} | "
               f"{o['sec_per_iteration'] / a['sec_per_iteration']:.2f}x | "
               f"{f['sec_per_iteration'] / a['sec_per_iteration']:.2f}x | "
               f"{M(o['env_steps_per_sec'])} | {M(f['env_steps_per_sec'])} |\n")
    return md + ("\n*One update per batch. The slowdown columns are against the same worker "
                 "running alone on the node.*\n\n")


def memory_saturation_table(d):
    """What the node's memory system delivers as more processes ask it for traffic at once."""
    rows = d.get("memory_saturation", [])
    if not rows:
        return ""
    md = ("| processes asking at once | total gigabytes per second | "
          "gigabytes per second each | share of the rate one process gets alone |\n"
          "|---|---|---|---|\n")
    solo = rows[0]["gb_per_sec_per_process"]
    for r in rows:
        md += (f"| {r['processes']} | {r['total_gb_per_sec']:,.1f} | "
               f"{r['gb_per_sec_per_process']:,.2f} | "
               f"{r['gb_per_sec_per_process'] / solo:.2f} |\n")
    return md + ("\n*Independent processes, each moving arrays far larger than any cache, run "
                 "with nothing else on the node.*\n\n")


def memory_corun_table(d):
    """What a stream of memory traffic gets while the training load runs beside it."""
    rows = d.get("memory_corun", [])
    if len(rows) < 2:
        return ""
    idle = rows[0]["stream_gb_per_sec_per_process"]
    md = ("| what else was running | gigabytes per second each stream process got | "
          "share of the idle-node rate |\n|---|---|---|\n")
    md += f"| nothing | {idle:,.2f} | 1.00 |\n"
    for r in rows[1:]:
        md += (f"| 216 training workers, {r['copies']} copies each | "
               f"{r['stream_gb_per_sec_per_process']:,.2f} | "
               f"{r['stream_gb_per_sec_per_process'] / idle:.2f} |\n")
    return md + ("\n*Eight stream processes, measured over twenty seconds after the training "
                 "load had been running for twenty-five. One update per batch.*\n\n")


def sec_correction():
    """Subsection: the two defects in the earlier processor measurements, and their size."""
    table = correction_table()
    if not table:
        return ""
    rows = correction_rows("jaguar03", 224, 16)
    worst = max(rows, key=lambda r: 1 - r["window"] / r["burst"])
    return f"""### 5.3 A correction to the processor numbers above

The processor measurements in this section were taken with a benchmark that times five
iterations, about two seconds of work, and computes the machine's rate as the sum of each
worker process's own rate. Both of those are wrong at large worker counts, for two separate
reasons.

The first is the processor. A server processor runs above its sustained clock for the first
seconds of a load and then settles, so a two-second measurement reads the opening burst rather
than the rate a training run of any length actually gets.

The second is the arithmetic. Adding up each process's own rate assumes every process was
running for the whole time the others were. Over five iterations, hundreds of processes are
still starting at staggered moments, so the sum describes a load that never existed on the
machine at one time. The fix is to hold every worker at a barrier until all of them have warmed
up, and then to count only the work done inside the wall-clock window in which every worker was
running.

{table}Both corrections point the same way and together they remove
**{100 * (1 - worst['window'] / worst['burst']):.0f}%** of the reported rate at this setting.
The middle column separates them: it is the old arithmetic applied to a settled load, so the
step from the first column to the second is the clock, and the step from the second to the third
is the overlap the old arithmetic assumed and did not have.

Every processor number in this section is now a sustained measurement over a shared window;
where a setting has not been retaken, its row says so.

"""


def best_worker_count(style):
    """Which worker count reaches the higher total throughput for one update convention.

    before: 224 workers peaking at 1.55 and 112 workers at 1.71 million steps per second
    after:  (112, its peak row, the 224 peak row)
    """
    peaks = {}
    for workers in (112, 224):
        rows = ladder(style, workers)
        if rows:
            peaks[workers] = max(rows, key=lambda r: r["env_steps_per_sec"])
    if not peaks:
        return None
    winner = max(peaks, key=lambda w: peaks[w]["env_steps_per_sec"])
    return winner, peaks


def what_ended_it(rows, flat):
    """Why the sweep stopped where it did for one series: a turnover, a flattening, or memory.

    before: rows whose totals rise to 1.88 at 64 copies then fall to 1.56 at 128
    after:  "turned over" — the curve is past its peak, not merely flat
    """
    top = max(rows, key=lambda r: r["env_steps_per_sec"])
    if rows[-1]["env_steps_per_sec"] < 0.98 * top["env_steps_per_sec"]:
        return "turned over"
    if flat:
        return "flattened"
    return "still rising where the sweep stopped"


def plateau_summary_table():
    """The headline of the sweep: one row per update convention and worker count.

    Every number a reader would compare or copy sits here rather than in the prose beside it.
    """
    rows = []
    for style, label in [("full_batch", "one update per batch"),
                         ("epoch_minibatch", "sixteen updates per batch")]:
        for workers in (112, 224):
            series = ladder(style, workers)
            if not series:
                continue
            top = max(series, key=lambda r: r["env_steps_per_sec"])
            rows.append({"label": label, "workers": workers, "top": top, "largest": series[-1],
                         "end": what_ended_it(series, plateau_rung(series))})
    if not rows:
        return ""
    md = ("| update convention | workers | best copies per worker | copies at the best setting | "
          "seconds per iteration | million steps per second | "
          "thousand steps per second per copy | hours per million steps per copy | "
          "peak memory per worker (GB) | what ended the sweep |\n"
          "|---|---|---|---|---|---|---|---|---|---|\n")
    for r in rows:
        t = r["top"]
        md += (f"| {r['label']} | {r['workers']} | {t['n_copies']} | {t['total_copies']:,} | "
               f"{t['sec_per_iteration']:.3f} | {M(t['env_steps_per_sec'])} | "
               f"{K(t['env_steps_per_sec_per_copy'])} | "
               f"{H(t['env_steps_per_sec_per_copy'])} | {GB(t['peak_rss_mb_max_worker'])} | "
               f"{r['end']} |\n")
    return md + ("\n*The best rung of each series, and what stopped the series there. Every rung "
                 "is in the four tables above.*\n\n")


def consequence_table():
    """What packing past the best rung costs, on both counts, for each update convention.

    The question the curve raises is whether there is ever a reason to put more copies in a
    worker than the best rung. The answer is in the two loss columns: past the peak the machine
    does less total work AND every copy in it runs slower, so nothing is traded for anything.
    """
    rows = []
    for style, label in [("full_batch", "one update per batch"),
                         ("epoch_minibatch", "sixteen updates per batch")]:
        found = best_worker_count(style)
        if not found:
            continue
        winner, _ = found
        series = ladder(style, winner)
        top = max(series, key=lambda r: r["env_steps_per_sec"])
        beyond = [r for r in series if r["n_copies"] > top["n_copies"]]
        if not beyond:
            continue
        rows.append({"label": label, "workers": winner, "top": top, "next": beyond[0],
                     "last": beyond[-1]})
    if not rows:
        return ""
    md = ("| update convention | workers | copies per worker | million steps per second | "
          "thousand steps per second per copy | hours per million steps per copy | "
          "against the best rung |\n|---|---|---|---|---|---|---|\n")
    for r in rows:
        for tag, row in [("the best rung", r["top"]), ("twice it", r["next"]),
                         ("the largest measured", r["last"])]:
            change = ("" if tag == "the best rung" else
                      f"{100 * (row['env_steps_per_sec'] / r['top']['env_steps_per_sec'] - 1):.0f}% "
                      f"total, "
                      f"{100 * (row['env_steps_per_sec_per_copy'] / r['top']['env_steps_per_sec_per_copy'] - 1):.0f}% "
                      f"per copy")
            md += (f"| {r['label']}, {tag} | {r['workers']} | {row['n_copies']} | "
                   f"{M(row['env_steps_per_sec'])} | {K(row['env_steps_per_sec_per_copy'])} | "
                   f"{H(row['env_steps_per_sec_per_copy'])} | {change} |\n")
    return md + ("\n*Both quantities fall together past the best rung, so packing more copies "
                 "into a worker buys nothing on either count.*\n\n")


def plateau_sentences(style, label):
    """The plateau, the peak and the memory for one update convention, all read from the rows."""
    found = best_worker_count(style)
    if not found:
        return ""
    winner, _ = found
    rows = ladder(style, winner)
    top = max(rows, key=lambda r: r["env_steps_per_sec"])
    end = what_ended_it(rows, plateau_rung(rows))
    if end == "turned over":
        meaning = (f"packing more than {top['n_copies']} copies into a worker takes throughput "
                   f"away rather than merely stopping to add it, so that rung is not a soft "
                   f"boundary but the setting to use")
    elif end == "flattened":
        meaning = (f"past {top['n_copies']} copies a worker the machine does no more work while "
                   f"every copy in it goes on getting slower, so nothing is bought by going "
                   f"further")
    else:
        meaning = ("the curve had not stopped rising where the sweep ended, so the setting to "
                   "use is the largest one measured and the ceiling is still above it")
    return (f"**{label}.** Read from the table: {meaning}. The node's memory is not what stops "
            f"it — at the best setting the whole node holds "
            f"{top['node_peak_used_gb']:,.0f} GB of its 1,008 GB.")


def sec_plateau():
    """Subsection: how far the copies per worker go, and what stops them."""
    n = plateau_numbers()
    if not n:
        return ""
    md = """### 5.4 How far the copies per worker go, and what stops them

The sweep above stops at 224 workers holding 16 copies each, and total throughput is still
rising there, so it does not show the machine's ceiling. Workers cannot be added — 224 is the
machine's logical-processor count — but each worker can hold more copies, so the ladder was
continued on that knob, at two worker counts: 224, which uses both hardware threads of every
core, and 112, which uses one thread per core and leaves the other idle.

Every rung below times about ninety seconds of continuous work with every worker synchronised,
as the correction above requires, and records the peak resident memory of its workers, since
copies per worker is what drives memory and the node has a fixed 1 TB of it.

"""
    for style, label in [("full_batch", "One update per batch"),
                         ("epoch_minibatch", "Sixteen updates per batch")]:
        for workers in (224, 112):
            key = (style, workers)
            if key in n:
                md += ladder_table(n[key]["rows"],
                                   f"{label}, {workers} independent single-thread workers.", style)
    md += "![copies per worker](figures/cpu_plateau.png)\n\n"
    md += plateau_summary_table()
    repeats = repeat_table()
    if repeats:
        md += ("Four settings were measured twice, to show how much a reading moves between two "
               "runs of the same thing.\n\n") + repeats
    for style, label in [("full_batch", "One update per batch"),
                         ("epoch_minibatch", "Sixteen updates per batch")]:
        sentence = plateau_sentences(style, label)
        if sentence:
            md += sentence + "\n\n"
    consequence = consequence_table()
    if consequence:
        md += ("**Is there a reason to pack more copies into a worker than that?** No, and the "
               "reason is that nothing is being traded. A setting past the peak is worse for the "
               "queue that cares about total throughput and worse for the run whose owner cares "
               "how long one copy takes.\n\n") + consequence
    return md


# the trainer's own tensors, counted once so the working-set arithmetic below is checkable:
# 59,910 parameters per copy held in four buffers (the parameters, their gradients and the two
# optimiser moments), a flattened batch of 512 rows whose nine tensors are 143 values wide in
# total and which is held twice (the batch and its shuffled twin), and the rollout buffers.
PARAMS_PER_COPY = 59910
ROWS_PER_COPY = 512
BATCH_VALUES_PER_ROW = 143
ROLLOUT_VALUES_PER_COPY = 9728
BYTES_PER_VALUE = 4
L3_BYTES_PER_SOCKET = 256 * 1024 * 1024
CORES_PER_SOCKET = 56


def working_set_bytes(copies):
    """The state one worker keeps for its copies, which is what its update walks each iteration.

    before: copies = 1
    after:  4 x 59,910 + 2 x 512 x 143 + 9,728 = 395,800 values, that is 1,583,200 bytes
    """
    per_copy = (4 * PARAMS_PER_COPY + 2 * ROWS_PER_COPY * BATCH_VALUES_PER_ROW
                + ROLLOUT_VALUES_PER_COPY)
    return copies * per_copy * BYTES_PER_VALUE


def working_set_table(copies_list=(1, 4, 16, 64, 128, 256, 512)):
    """What a worker keeps, against the share of the last-level cache one core has."""
    share = L3_BYTES_PER_SOCKET / CORES_PER_SOCKET
    md = ("| copies per worker | state the worker keeps (MB) | times one core's share of the "
          "last-level cache |\n|---|---|---|\n")
    for c in copies_list:
        b = working_set_bytes(c)
        md += f"| {c} | {b / 1e6:,.1f} | {b / share:,.1f} |\n"
    return md + (f"\n*One core's share of the last-level cache is {share / 1e6:.1f} MB "
                 f"({L3_BYTES_PER_SOCKET // (1024 * 1024)} MB per socket over {CORES_PER_SOCKET} "
                 f"cores). At 224 workers two workers share one core, so the demand on that share "
                 f"is twice the figure in the last column.*\n\n")


def sec_cause():
    """Subsection: which resource ran out at the plateau, and the measurements that say so."""
    d = plateau()
    if not d:
        return ""
    tables = (contention_table(d) + memory_saturation_table(d) + memory_corun_table(d))
    if not tables.strip():
        return ""
    share = L3_BYTES_PER_SOCKET / CORES_PER_SOCKET
    md = f"""### 5.5 What ran out

Three measurements, each isolating one candidate.

The first is competition. The same worker, holding the same copies, was run alone on the empty
node, then as one of 112, then as one of 224. A worker alone has the whole memory system to
itself, so the gap between its seconds per iteration and the same worker's inside a full machine
is what competition costs; the gap between 112 workers and 224 is what sharing a core with a
second hardware thread costs on top of that.

{contention_table(d)}The second is the memory system's own rate. Independent processes moving
arrays far larger than any cache were run on the empty node, one to 224 of them, and their summed
rate is what the memory system delivers.

{memory_saturation_table(d)}The third puts the two together: the same stream processes measured
while the training load ran beside them. If the training load has the memory system full, the
stream processes get a small fraction of what they get on an idle node.

{memory_corun_table(d)}The arithmetic behind all three is the size of what a worker walks. One
copy keeps the parameters, their gradients, the two optimiser moments, the flattened batch and its
shuffled twin, and the rollout buffers, and the update touches all of it every iteration. The
last-level cache is {L3_BYTES_PER_SOCKET // (1024 * 1024)} MB per socket shared by
{CORES_PER_SOCKET} cores, that is {share / 1e6:.1f} MB a core.

{working_set_table()}From four copies upward a worker keeps more than its share of the cache, so
the update is reading from memory rather than from cache, and the copies-per-worker knob is really
a knob on how much memory traffic each core makes.

The two update conventions differ by exactly that. Sixteen updates per batch reads and writes the
parameter-side buffers sixteen times per iteration where one update per batch does it once, for
the same 512 environment steps per copy, so it asks the memory system for about sixteen times the
parameter traffic per environment step — which is why it is the slower convention everywhere in
the tables above and why it turns over at a quarter of the copies per worker.

"""
    return md


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
        md += process_table(tr_proc, "Independent single-thread processes. Sixteen updates per "
                                     "batch.")
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
    md += sec_correction()
    md += sec_plateau()
    md += sec_cause()
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
actually operates in. For every platform and configuration but one, the table gives the setting
that reaches the highest total throughput inside that range; the exception is the
one-copy-per-worker processor row, explained under the table.

| platform and configuration | copies | seconds per iteration | million steps per second ↑ | thousand steps per second per copy | hours per million steps per copy |
|---|---|---|---|---|---|
"""
    # mark best and second best in every scored column (copies is a setting, so it is skipped);
    # hours sorts the opposite way from the two rates, since fewer hours is better
    cells = {"sec_per_iteration": [f"{r['sec_per_iteration']:.3f}" for r in rows],
             "total": [M(r["total"]) for r in rows],
             "per_copy": [K(r["per_copy"]) for r in rows],
             "hours_1M": [f"{r['hours_1M']:.3g}" for r in rows]}
    for field, higher_is_better in [("sec_per_iteration", False), ("total", True),
                                    ("per_copy", True), ("hours_1M", False)]:
        cells[field] = mark_best(cells[field], [r[field] for r in rows], higher_is_better)
    for i, r in enumerate(rows):
        md += (f"| {r['name']} | {r['copies']:,} | {cells['sec_per_iteration'][i]} | "
               f"{cells['total'][i]} | {cells['per_copy'][i]} | {cells['hours_1M'][i]} |\n")
    gpu = [r for r in rows if r["kind"] == "gpu"]
    cpu = [r for r in rows if r["kind"] == "cpu"]
    # the pinned row is in the table for the rate one copy gets, so say what it costs against the
    # processor row that wins on total; both sides are read from the rows the table just printed
    solo = [r for r in rows if r["pinned"]][0]
    quickest = max(cpu, key=lambda r: r["per_copy"])
    if quickest is solo:
        md += (f"\nThe {solo['copies']:,}-copy row is the exception: it is the processor setting "
               f"that finishes any single copy soonest, giving each copy {K(solo['per_copy'])} "
               f"thousand steps per second against {K(cpu[0]['per_copy'])} thousand for the "
               f"processor setting that wins on total throughput — a million steps per copy in "
               f"{solo['hours_1M']:.3f} hours instead of {cpu[0]['hours_1M']:.3f} — at the price "
               f"of a factor of {cpu[0]['total'] / solo['total']:.1f} in total throughput.\n")
    else:
        # measured over a long window rather than a two-second one, giving every worker a single
        # copy stopped being the way to finish one copy soonest: the batched settings beat it on
        # both counts, and the row stays in the table to show that
        md += (f"\nThe {solo['copies']:,}-copy row is in the table for a claim that the sustained "
               f"measurements withdrew. Giving every worker one copy was the setting that "
               f"finished a single copy soonest when these numbers were read off two-second "
               f"measurements. Measured over a long window it is not: it gives each copy "
               f"{K(solo['per_copy'])} thousand steps per second, where "
               f"{quickest['name']} at {quickest['copies']:,} copies gives "
               f"{K(quickest['per_copy'])} thousand — a million steps per copy in "
               f"{quickest['hours_1M']:.3f} hours against {solo['hours_1M']:.3f} — while also "
               f"reaching {quickest['total'] / solo['total']:.0f} times its total throughput. "
               f"Packing copies into each worker is not a trade against single-copy speed on this "
               f"machine; up to the plateau it is better at both.\n")
    md += "\n![best setup](figures/best_setup.png)\n\n"
    if gpu and cpu:
        ratio = gpu[0]["total"] / cpu[0]["total"]
        md += (f"The best graphics-processor configuration reaches {M(gpu[0]['total'])} million "
               f"environment steps per second at {gpu[0]['copies']:,} copies; the best processor "
               f"configuration reaches {M(cpu[0]['total'])} million at {cpu[0]['copies']:,} "
               f"copies. That is a factor of **{ratio:.1f}**. In wall-clock terms, giving every "
               f"copy a million environment steps takes {gpu[0]['hours_1M']:.3f} hours on the "
               f"graphics processor against {cpu[0]['hours_1M']:.3f} hours on the processor "
               f"node.\n\n")
    md += """Best in each column is bold, second best underlined; copies is a setting rather than
a score, so it is not marked. Two qualifications belong with those numbers. The processor figure is for one node held
exclusively; a cluster with many such nodes multiplies it, and the independent-process
arrangement is exactly what a work queue across many nodes would do. And a configuration that
wins on total throughput is not the one that finishes any single copy soonest, which is why both
rates appear in every table and both curves in every figure.

"""
    return md


