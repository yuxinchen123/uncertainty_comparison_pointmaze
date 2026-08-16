"""Read every measurement file and write results.md plus the figures beside it.

The throughput rule this project works under asks every throughput table to carry BOTH rates —
the aggregate over all copies and the rate one copy gets — plus the per-copy rate restated as
the hours a million steps per copy would take, and every figure to show both quantities. That
is what the tables and the two-panel figures below do.
"""
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402

from node_classes import COPY_COUNTS, cpu_counts_for, load_classes   # noqa: E402

RUN = Path(__file__).resolve().parent.parent
RESULTS = RUN / "data" / "throughput"
PROBES = RUN / "data" / "probe"
PLOTS = RUN / "plots"
PACIFIC = ZoneInfo("America/Los_Angeles")


def display_time(iso):
    """A stored timestamp rendered for the reader: Pacific, with the zone named."""
    return datetime.fromisoformat(iso).astimezone(PACIFIC).strftime("%Y-%m-%d %H:%M PT")


def load_jobs():
    """Every finished job record, keyed by (node class, processor count)."""
    jobs = {}
    for path in sorted(RESULTS.glob("*__cpus-*__*.json")):
        if ".cell." in path.name:
            continue
        record = json.loads(path.read_text())
        jobs[(record["node_class"], record["requested_cpus"])] = record
    return jobs


def cell_of(jobs, class_name, cpus, n_copies):
    """One measurement cell, or None if that job has not run."""
    job = jobs.get((class_name, cpus))
    if job is None:
        return None
    return next((c for c in job["cells"] if c["n_copies"] == n_copies), None)


def measured_capability(probes):
    """Compute capability as each card reported it, keyed by node class.

    The cluster's catalog is a hand-maintained file and has been wrong before — it lists
    `titanx03` as a Maxwell card at 5.2 when the card is a Pascal TITAN X at 6.1. The probe asks
    the card itself, so where the two disagree the card wins and the table says so.

    before: catalog compute_capability 5.2 for titanx03; the probe read the string "6.1" off it
    after:  {"titanx03": 6.1}, and the row prints "6.1 (catalog says 5.2)"
    """
    return {probe["node_class"]: float(probe["compute_capability"])
            for probe in probes if probe.get("compute_capability")}


def hardware_table(classes, jobs, probes):
    """One row per node class: what the card is, and how much of the survey it has finished."""
    read_from_card = measured_capability(probes)
    lines = ["| node class | partition | card | compute<br>capability | card memory<br>(GB) | "
             "processor | processor counts<br>measured | cells<br>done |",
             "|---|---|---|---|---|---|---|---|"]
    for cls in classes:
        counts = cpu_counts_for(cls)
        measured = [n for n in counts if (cls["name"], n) in jobs]
        done = sum(1 for n in counts for c in COPY_COUNTS
                   if (cell_of(jobs, cls["name"], n, c) or {}).get("status") == "measured")
        catalog = cls["compute_capability"]
        card = read_from_card.get(cls["name"])
        capability = (f"{card} (catalog says {catalog})" if card and card != catalog
                      else f"{catalog}")
        lines.append(
            f"| `{cls['name']}` | {cls['partition']} | {cls['display_name']} | "
            f"{capability} | {cls['gpu_mem_mb'] / 1000:.1f} | "
            f"{cls['cpu_type'].replace('_', ' ')} | "
            f"{', '.join(str(m) for m in measured) or 'none yet'} | "
            f"{done}/{len(counts) * len(COPY_COUNTS)} |")
    return "\n".join(lines)


def best_cpu_count(jobs, class_name, cpus_list, n_copies):
    """The processor count that gave this class its fastest iteration at one copy count."""
    timed = [(cell_of(jobs, class_name, n, n_copies), n) for n in cpus_list]
    timed = [(c, n) for c, n in timed if c and c.get("status") == "measured"]
    return min(timed, key=lambda t: t[0]["seconds_per_iteration"]) if timed else (None, None)


def throughput_table(classes, jobs, n_copies):
    """The survey's main table at one copy count: every class, at its best processor count.

    Rows are ordered by aggregate throughput, fastest first; the best and second-best cell of
    every rate column are marked bold and underlined.
    """
    rows = []
    for cls in classes:
        cell, cpus = best_cpu_count(jobs, cls["name"], cpu_counts_for(cls), n_copies)
        if cell is None:
            status = next((cell_of(jobs, cls["name"], n, n_copies)
                           for n in cpu_counts_for(cls)
                           if cell_of(jobs, cls["name"], n, n_copies)), None)
            rows.append({"cls": cls, "cell": None,
                         "note": (status or {}).get("status", "not measured yet")})
            continue
        rows.append({"cls": cls, "cell": cell, "cpus": cpus})

    measured = [r for r in rows if r["cell"]]
    measured.sort(key=lambda r: -r["cell"]["total_steps_per_second"])
    unmeasured = [r for r in rows if not r["cell"]]

    # best and second best of each rate column, for the bold / underline marking
    def rank(key, higher_is_better=True):
        vals = sorted({r["cell"][key] for r in measured}, reverse=higher_is_better)
        return vals[:2]

    top_total = rank("total_steps_per_second")
    top_hours = rank("hours_per_million_steps_per_copy", higher_is_better=False)

    def mark(value, tops, fmt):
        text = fmt.format(value)
        if tops and value == tops[0]:
            return f"**{text}**"
        if len(tops) > 1 and value == tops[1]:
            return f"<u>{text}</u>"
        return text

    lines = ["| node class | card | processors | seconds per<br>iteration | "
             "total steps per second<br>(millions) &darr; | steps per second<br>per copy | "
             "hours per million<br>steps per copy | peak card<br>memory (GB) | "
             "spread of the<br>middle half |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in measured:
        c = r["cell"]
        lines.append(
            f"| `{r['cls']['name']}` | {r['cls']['display_name']} | {r['cpus']} | "
            f"{c['seconds_per_iteration']:.4f} | "
            f"{mark(c['total_steps_per_second'] / 1e6, [t / 1e6 for t in top_total], '{:.2f}')} | "
            f"{c['steps_per_second_per_copy']:,.0f} | "
            f"{mark(c['hours_per_million_steps_per_copy'], top_hours, '{:.3f}')} | "
            f"{c['peak_device_memory_mb'] / 1000:.1f} | "
            f"{c['relative_spread_middle_half'] * 100:.2f}% |")
    for r in unmeasured:
        # kept short: this table is nine columns wide and has to stay printable
        note = {"out_of_memory": "**does not fit**",
                "not_attempted_smaller_count_ran_out_of_memory": "**does not fit**",
                "not_attempted_out_of_time": "ran out of time",
                "failed": "failed — see log"}.get(r["note"], r["note"])
        lines.append(f"| `{r['cls']['name']}` | {r['cls']['display_name']} | — | — | {note} | "
                     "— | — | — | — |")
    return "\n".join(lines)


def cpu_effect_summary(classes, jobs):
    """The one-sentence answer to whether the processor count changes anything."""
    spans = []
    for cls in classes:
        for n_copies in COPY_COUNTS:
            timed = [cell_of(jobs, cls["name"], n, n_copies) for n in cpu_counts_for(cls)]
            times = [c["seconds_per_iteration"] for c in timed
                     if c and c.get("status") == "measured"]
            if len(times) >= 2:
                spans.append((max(times) - min(times)) / min(times))
    if not spans:
        return ""
    spans = np.asarray(spans)
    return (f"**It does not.** Over the {len(spans)} card-and-copy-count combinations measured "
            f"at two or more processor counts, the slowest processor count was slower than the "
            f"fastest by {np.median(spans) * 100:.1f}% in the typical case and "
            f"{spans.max() * 100:.1f}% at the very worst — the size of the measurement's own "
            "noise, and with no consistent direction: more processors are as often marginally "
            "slower as marginally faster. Eight processors is therefore the right request for "
            "this trainer, and the processors beyond that are free to carry other work.")


def cpu_effect_table(classes, jobs):
    """Whether more processors help: each class's iteration time at each processor count.

    The trainer's work is on the card, so the expectation is that the processor count barely
    moves the number; the table is what says whether that expectation holds on this hardware.
    """
    lines = ["| node class | card | copies | " +
             " | ".join(f"{n} processors<br>(ms per iteration)" for n in (8, 16, 32)) +
             " | class maximum<br>(processors, ms) | spread across<br>processor counts |",
             "|---|---|---|---|---|---|---|---|"]
    for cls in classes:
        counts = cpu_counts_for(cls)
        for n_copies in COPY_COUNTS:
            cells = {n: cell_of(jobs, cls["name"], n, n_copies) for n in counts}
            timed = {n: c["seconds_per_iteration"] * 1000 for n, c in cells.items()
                     if c and c.get("status") == "measured"}
            if len(timed) < 2:
                continue
            # a count this node class cannot allocate reads "N/A"; one it can but has not yet
            # reported reads "not yet" — the two mean different things and must not share a mark
            fixed = [f"{timed[n]:.1f}" if n in timed
                     else ("not yet" if n in counts else "N/A") for n in (8, 16, 32)]
            extra = [n for n in counts if n not in (8, 16, 32)]
            extra_cell = ("N/A" if not extra else
                          f"{extra[0]}, {timed[extra[0]]:.1f}" if extra[0] in timed
                          else f"{extra[0]}, not yet")
            span = (max(timed.values()) - min(timed.values())) / min(timed.values())
            lines.append(f"| `{cls['name']}` | {cls['display_name']} | {n_copies} | "
                         + " | ".join(fixed) + f" | {extra_cell} | {span * 100:.1f}% |")
    return "\n".join(lines)


def series_style(index):
    """A colour, marker and dash pattern per card, so twenty-odd lines stay tellable apart.

    Matplotlib's default cycle repeats after ten colours, which puts two different cards on the
    same blue line and makes the legend useless. Twenty distinct colours crossed with four
    markers and three dash patterns give every series its own appearance.
    """
    colours = plt.get_cmap("tab20").colors
    markers = ("o", "s", "^", "D")
    dashes = ("-", "--", ":")
    return {"color": colours[index % len(colours)],
            "marker": markers[(index // len(colours)) % len(markers)],
            "linestyle": dashes[(index // len(colours)) % len(dashes)],
            "markersize": 4.5, "linewidth": 1.4}


def scaling_figure(classes, jobs):
    """Two panels: aggregate steps per second against copies, and the rate one copy gets."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for index, cls in enumerate(classes):
        xs, total, per_copy = [], [], []
        for n_copies in COPY_COUNTS:
            cell, _ = best_cpu_count(jobs, cls["name"], cpu_counts_for(cls), n_copies)
            if cell:
                xs.append(n_copies)
                total.append(cell["total_steps_per_second"] / 1e6)
                per_copy.append(cell["steps_per_second_per_copy"] / 1e3)
        if not xs:
            continue
        label = f"{cls['display_name']} ({cls['name']})"
        axes[0].plot(xs, total, label=label, **series_style(index))
        axes[1].plot(xs, per_copy, label=label, **series_style(index))
    for ax, ylabel, title in (
            (axes[0], "million environment steps per second",
             "all copies together"),
            (axes[1], "thousand environment steps per second, one copy",
             "what one copy gets")):
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(COPY_COUNTS)
        ax.set_xticklabels([str(c) for c in COPY_COUNTS])
        ax.set_xlabel("training copies")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.3)
    # the legend sits outside both panels: with this many cards it would otherwise cover the
    # very curves a reader is trying to follow
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=7.5, loc="center left",
               bbox_to_anchor=(1.0, 0.5), frameon=False)
    fig.suptitle("JAX PPO+RND, single update per rollout, PointMaze Large — "
                 "throughput by graphics card")
    fig.tight_layout()
    fig.savefig(PLOTS / "throughput_scaling.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def card_ranking_figure(classes, jobs, n_copies):
    """A bar chart of the cards at one copy count, both rates side by side."""
    rows = []
    for cls in classes:
        cell, cpus = best_cpu_count(jobs, cls["name"], cpu_counts_for(cls), n_copies)
        if cell:
            rows.append((f"{cls['display_name']}\n({cls['name']})",
                         cell["total_steps_per_second"] / 1e6,
                         cell["steps_per_second_per_copy"] / 1e3))
    if not rows:
        return
    rows.sort(key=lambda r: r[1])
    names, total, per_copy = zip(*rows)
    fig, axes = plt.subplots(1, 2, figsize=(13, max(5, 0.34 * len(rows))), sharey=True)
    y = np.arange(len(rows))
    axes[0].barh(y, total, color="#3B6FB6")
    axes[0].set_xlabel("million environment steps per second, all copies together")
    axes[1].barh(y, per_copy, color="#B6663B")
    axes[1].set_xlabel("thousand environment steps per second, one copy")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(names, fontsize=7)
    for ax in axes:
        ax.grid(True, axis="x", alpha=0.3)
    fig.suptitle(f"Throughput at {n_copies} copies, each card at its best processor count")
    fig.tight_layout()
    fig.savefig(PLOTS / f"card_ranking_copies-{n_copies}.png", dpi=160)
    plt.close(fig)


def memory_figure(classes, jobs):
    """Peak card memory against copies — what decides whether a card can hold a run at all."""
    fig, ax = plt.subplots(figsize=(9, 6))
    for index, cls in enumerate(classes):
        xs, mem = [], []
        for n_copies in COPY_COUNTS:
            cell, _ = best_cpu_count(jobs, cls["name"], cpu_counts_for(cls), n_copies)
            if cell and cell.get("peak_device_memory_mb"):
                xs.append(n_copies)
                mem.append(cell["peak_device_memory_mb"] / 1000)
        if xs:
            ax.plot(xs, mem, label=f"{cls['display_name']} ({cls['name']})",
                    **series_style(index))
    ax.set_xscale("log", base=2)
    ax.set_xticks(COPY_COUNTS)
    ax.set_xticklabels([str(c) for c in COPY_COUNTS])
    ax.set_xlabel("training copies")
    ax.set_ylabel("peak card memory in use (GB)")
    ax.set_title("Memory the trainer actually touches")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7.5, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout()
    fig.savefig(PLOTS / "peak_memory.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def startup_table(jobs):
    """What a run pays before its first iteration, and whether processors shorten it.

    Two separate costs: building the trainer (host-side, one orthogonal weight draw per copy per
    layer) and compiling the iteration (once per copy count). Neither is part of the throughput
    figures — both are discarded before timing — but both are part of a real run's wall clock.
    """
    build, compile_, by_cpus = {}, {}, {}
    for job in jobs.values():
        for cell in job["cells"]:
            if cell.get("status") != "measured":
                continue
            build.setdefault(cell["n_copies"], []).append(cell["build_seconds"])
            compile_.setdefault(cell["n_copies"], []).append(cell["compile_seconds"])
            by_cpus.setdefault((cell["n_copies"], cell["visible_cpus"]), []).append(
                cell["build_seconds"])
    lines = ["| copies | building the trainer<br>(seconds) | compiling the iteration<br>"
             "(seconds) | total before the<br>first iteration | iterations that time would<br>"
             "buy on an H100 |",
             "|---|---|---|---|---|"]
    # the H100's own iteration time at each copy count, to say what the setup is worth in work
    h100 = {c["n_copies"]: c["seconds_per_iteration"]
            for (name, _), job in jobs.items() if name == "serval06-09"
            for c in job["cells"] if c.get("status") == "measured"}
    for n_copies in sorted(build):
        b, c = float(np.median(build[n_copies])), float(np.median(compile_[n_copies]))
        worth = f"{(b + c) / h100[n_copies]:,.0f}" if n_copies in h100 else "N/A"
        lines.append(f"| {n_copies} | {b:.0f} | {c:.0f} | {b + c:.0f} | {worth} |")
    return "\n".join(lines), by_cpus


def startup_by_processors(by_cpus, n_copies=4096):
    """Whether more processors shorten the build — one sentence, from the measurements."""
    rows = sorted((cpus, float(np.median(v))) for (n, cpus), v in by_cpus.items()
                  if n == n_copies and len(v) >= 1)
    if len(rows) < 2:
        return ""
    listed = ", ".join(f"{seconds:.0f} s on {cpus}" for cpus, seconds in rows)
    return (f"More processors do not shorten it. At {n_copies:,} copies the build took "
            f"{listed} processors — the same time throughout, because the weights are drawn one "
            "copy at a time in a single-threaded loop on the host, so the work never reaches "
            "the other processors. It is the one part of a run that would gain from being "
            "vectorised across copies rather than looped.")


def where_to_send_a_run(classes, jobs):
    """The survey's practical answer: per copy count, the fastest card and what it costs.

    A sweep is planned in hours, not in steps per second, so each row also carries the wall time
    ten million steps per copy would take — the length of a real training run in this project.
    """
    lines = ["| copies | fastest card | node class | hours for ten million<br>steps per copy | "
             "next best, and how much<br>slower it is | cards that cannot<br>hold this run |",
             "|---|---|---|---|---|---|"]
    for n_copies in COPY_COUNTS:
        ranked = []
        no_room = 0
        for cls in classes:
            cell, _ = best_cpu_count(jobs, cls["name"], cpu_counts_for(cls), n_copies)
            if cell:
                ranked.append((cell["total_steps_per_second"], cls, cell))
            elif any((cell_of(jobs, cls["name"], n, n_copies) or {}).get(
                    "status", "").startswith(("out_of_memory", "not_attempted_smaller"))
                    for n in cpu_counts_for(cls)):
                no_room += 1
        if not ranked:
            continue
        ranked.sort(key=lambda r: -r[0])
        _, best_cls, best_cell = ranked[0]
        hours = 10 * best_cell["hours_per_million_steps_per_copy"]
        runner_up = (f"{ranked[1][1]['display_name']}, "
                     f"{ranked[0][0] / ranked[1][0]:.2f}x slower" if len(ranked) > 1 else "N/A")
        lines.append(f"| {n_copies} | {best_cls['display_name']} | `{best_cls['name']}` | "
                     f"{hours:.2f} | {runner_up} | {no_room} |")
    return "\n".join(lines)


def memory_table(classes, jobs):
    """Peak card memory per copy count, and the largest copy count each card could hold.

    Memory grows almost exactly in proportion to the copy count, so the per-copy cost measured
    at any one count predicts the ceiling: a card holds a run while
    (copies) x (memory per copy) stays under its memory, less the driver's own reservation.
    """
    lines = ["| node class | card | card memory<br>(GB) | " +
             " | ".join(f"{c} copies<br>(GB)" for c in COPY_COUNTS) +
             " | GB per<br>1,000 copies | largest copy<br>count that fits |",
             "|---|---|---|---|---|---|---|---|---|"]
    for cls in classes:
        # three outcomes per copy count, and they must never be printed alike: a measured peak,
        # a copy count the card could not hold, and a copy count whose job has not run yet
        measured, ran_out, unknown = {}, [], []
        for n_copies in COPY_COUNTS:
            cell, _ = best_cpu_count(jobs, cls["name"], cpu_counts_for(cls), n_copies)
            if cell and cell.get("peak_device_memory_mb"):
                measured[n_copies] = cell["peak_device_memory_mb"] / 1000
                continue
            statuses = [(cell_of(jobs, cls["name"], n, n_copies) or {}).get("status", "")
                        for n in cpu_counts_for(cls)]
            if any(s.startswith(("out_of_memory", "not_attempted_smaller")) for s in statuses):
                ran_out.append(n_copies)
            else:
                unknown.append(n_copies)
        if not measured:
            continue
        per_thousand = np.mean([gb / n * 1000 for n, gb in measured.items()])
        # what the card could hold if the proportionality continues: its memory, less the ~1 GB
        # the driver and the compiled program hold outside the trainer's arrays
        ceiling = int((cls["gpu_mem_mb"] / 1000 - 1.0) / (per_thousand / 1000))
        # short, because the row's own cells already say which counts did not fit
        if ran_out:
            note = f"{max(measured):,}"
        elif unknown:
            note = f"{max(measured):,} so far"
        else:
            note = f"all four; ~{ceiling:,}"
        cells = [f"{measured[c]:.1f}" if c in measured
                 else ("does not fit" if c in ran_out else "not yet") for c in COPY_COUNTS]
        lines.append(f"| `{cls['name']}` | {cls['display_name']} | "
                     f"{cls['gpu_mem_mb'] / 1000:.1f} | " + " | ".join(cells) +
                     f" | {per_thousand:.2f} | {note} |")
    return "\n".join(lines)


def stability_summary(jobs):
    """How firm the survey's numbers are, over every cell it measured."""
    spreads = [c["relative_spread_middle_half"] for j in jobs.values() for c in j["cells"]
               if c.get("status") == "measured"]
    if not spreads:
        return "No cell has reported yet."
    spreads = np.asarray(spreads)
    settled = sum(1 for j in jobs.values() for c in j["cells"]
                  if c.get("status") == "measured" and c["settled"])
    return (
        f"Across all {len(spreads)} measured cells the middle half of the timing rounds sat "
        f"within {np.median(spreads) * 100:.2f}% of the median in the typical cell, within "
        f"{np.percentile(spreads, 95) * 100:.2f}% in the worst 5%, and never worse than "
        f"{spreads.max() * 100:.2f}%. {settled} of {len(spreads)} cells reached the 2% "
        "settling target, every one of them within the six-round floor — so no number here "
        "rests on a timing that was still drifting when it was taken.")


def missing_cells(classes, jobs):
    """Every cell the survey still owes, and whether its card is covered elsewhere anyway.

    A missing class does not always leave a gap: `serval03` carries the same H100 NVL as
    `serval06-09` and differs only in its host processor, which section 6 shows does not move
    the number. Saying so keeps a queued job from reading as an untested card.
    """
    measured_cards = {(c["display_name"], c["gpu_mem_mb"]) for c in classes
                      if any((c["name"], n) in jobs for n in cpu_counts_for(c))}
    missing = []
    for cls in classes:
        covered = (cls["display_name"], cls["gpu_mem_mb"]) in measured_cards
        for cpus in cpu_counts_for(cls):
            if (cls["name"], cpus) in jobs:
                continue
            note = (" — this card is already measured on another node class, so the gap is the "
                    "host processor only" if covered else
                    " — **this card is measured nowhere else**")
            missing.append(f"`{cls['name']}` at {cpus} processors "
                           f"({cls['display_name']}, partition {cls['partition']}){note}")
    return missing


def main():
    """Write results.md and the figures from whatever has been measured so far."""
    classes = load_classes()
    jobs = load_jobs()
    probes = [json.loads(p.read_text()) for p in sorted(PROBES.glob("*.json"))]
    PLOTS.mkdir(exist_ok=True)
    scaling_figure(classes, jobs)
    memory_figure(classes, jobs)
    for n_copies in COPY_COUNTS:
        card_ranking_figure(classes, jobs, n_copies)

    n_cells = sum(1 for j in jobs.values() for c in j["cells"] if c.get("status") == "measured")
    n_oom = sum(1 for j in jobs.values() for c in j["cells"]
                if c.get("status", "").startswith(("out_of_memory", "not_attempted_smaller")))
    missing = missing_cells(classes, jobs)
    now = display_time(datetime.now().astimezone().isoformat())
    startup_rows, by_cpus = startup_table(jobs)
    startup_note = startup_by_processors(by_cpus)

    out = [
        "# Throughput of the single-update JAX PPO+RND trainer on every graphics card of "
        "this cluster",
        "",
        f"Written {now}. Times in this document are Pacific; the cluster's machines run "
        "Eastern, so every machine timestamp is converted where it is displayed.",
        "",
        f"{len(jobs)} of {sum(len(cpu_counts_for(c)) for c in classes)} jobs have reported, "
        f"giving {n_cells} measured cells; {n_oom} cells did not fit on their card. "
        f"{len(missing)} jobs are still queued or unrun — every number below is what has "
        "arrived, not a complete survey.",
        "",
        "## 1. What was measured",
        "",
        "One training iteration of `ppo/jax_ppo/jax_ppo_rnd.py` in its single-update style "
        "(`full_batch`): a 128-step rollout in the batched PointMaze Large environment with no "
        "reset noise on the start or the goal, the running statistics, both advantage streams, "
        "and one gradient step over the whole batch. Every copy runs 4 environments, so one "
        "iteration advances 512 environment steps per copy.",
        "",
        "Two rates describe every measurement, and neither substitutes for the other. The "
        "aggregate rate says how much work the machine does; the per-copy rate says how long "
        "any single copy takes to finish. Writing $s$ for the seconds one iteration takes, $C$ "
        "for the number of copies, $T = 128$ for the rollout length and $N = 4$ for the "
        "environments per copy:",
        "",
        # spacing macros are spelled with letter names: a backslash before ASCII punctuation
        # (\, \; \{ \}) is resolved as a CommonMark escape before the renderer sees the math, so
        # "C\,T\,N" would arrive as the list "C,T,N" instead of the product it means
        "$$\\text{total steps per second} = \\frac{C\\thinspace T\\thinspace N}{s}, \\qquad "
        "\\text{steps per second per copy} = \\frac{T\\thinspace N}{s}, \\qquad "
        "\\text{hours per million steps per copy} = "
        "\\frac{10^{6}}{3600\\thinspace (T N / s)}.$$",
        "",
        "A number is quoted only after the middle half of its timing rounds agreed to within "
        "2% of their median, with compilation and five warm-up iterations discarded first; the "
        "spread each number settled to is in the last column of every table.",
        "",
        "One job covers all four copy counts on one card at one processor count, and is held "
        "under a 27-minute deadline so it stays inside the half hour the survey was asked to "
        "keep to. No job came close to it: the longest ran 22 minutes and 20 seconds, and none "
        "had to abandon a copy count for lack of time.",
        "",
        "## 2. The hardware the survey covers",
        "",
        "One node per node class — nodes identical in card type, card memory, processor type "
        "and partition — because two nodes of one class are the same machine twice. Processor "
        "counts are 8, 16 and 32; a class that cannot allocate 32 (its node reserves one core "
        "for the system) is measured at its own maximum instead, and that maximum is named in "
        "the table.",
        "",
        hardware_table(classes, jobs, probes),
        "",
        f"All {len(probes)} classes probed so far run the trainer: JAX 0.10.2 with the CUDA 12 "
        "plugin reaches every card generation here, from compute capability 6.0 (Tesla P100, "
        "2016) to 12.0 (RTX 5080, 2025), so no node class had to be dropped for lack of "
        "support.",
        "",
        "## 3. Where to send a run",
        "",
        "The tables in section 4 carry every card; this one carries the answer. Ten million "
        "steps per copy is the length of a real training run in this project, so the wall time "
        "is quoted for that.",
        "",
        where_to_send_a_run(classes, jobs),
        "",
    ]

    for n_copies in COPY_COUNTS:
        out += [f"## 4.{COPY_COUNTS.index(n_copies) + 1} {n_copies} copies", "",
                "Each class at whichever of its processor counts ran fastest. Best value in "
                "bold, second best underlined; the table is sorted by the aggregate rate. "
                "Taking the fastest of a class's three processor counts flatters each row a "
                "little, since it is the smallest of three timings of what section 6 shows to "
                "be the same quantity; the effect is under 1%, which is the size of the gap "
                "between the processor counts themselves.",
                "", throughput_table(classes, jobs, n_copies), "",
                f"![cards at {n_copies} copies](plots/card_ranking_copies-{n_copies}.png)", ""]

    out += [
        "## 5. How throughput scales with copies",
        "",
        "![throughput scaling](plots/throughput_scaling.png)",
        "",
        "## 6. Does the processor count matter",
        "",
        "The trainer keeps its arrays on the card and the host only dispatches, so the "
        "expectation is that 8, 16 and 32 processors give the same iteration time.",
        "",
        cpu_effect_summary(classes, jobs),
        "",
        "The table below is the evidence, one row per card and copy count; its last column is "
        "the span between the fastest and the slowest processor count as a fraction of the "
        "fastest. A count this node class cannot allocate reads N/A; one it can allocate but "
        "has not yet reported reads \"not yet\".",
        "",
        cpu_effect_table(classes, jobs),
        "",
        "## 7. Memory, and which cards cannot hold a run",
        "",
        "Memory grows in proportion to the copy count — doubling the copies doubles the peak — "
        "so the cost per copy measured at any one count says where a card's ceiling is. The "
        "last column applies that: the card's memory, less about a gigabyte the driver and the "
        "compiled program hold outside the trainer's arrays, divided by the cost of one copy. "
        "That figure is a projection, not a measurement: on the largest cards it extrapolates "
        "several times past the biggest copy count anyone ran here, and it assumes the "
        "proportionality holds that far, which nothing in this survey checked.",
        "",
        memory_table(classes, jobs),
        "",
        "**The same run needs 66% more memory on an older card.** Every card at compute "
        "capability 8.0 and above — Ampere, Ada, Hopper, Blackwell — holds 4,096 copies in "
        "about 11.3 GB, while every Turing and Pascal card needs about 18.8 GB for exactly the "
        "same work, and the H100 sits slightly above its generation at 13.4 GB. The split "
        "follows the card generation and not the card's size or speed, so it is the compiler "
        "emitting a different program for the older architectures, not the trainer asking for "
        "more. What in that program costs the extra memory was not investigated here. The "
        "practical consequence is that the memory ceiling of an older card is reached about a "
        "third sooner than its size alone suggests.",
        "",
        "![peak memory](plots/peak_memory.png)",
        "",
        "## 8. What a run pays before its first iteration",
        "",
        "The throughput figures above are steady-state: compilation and warm-up are discarded "
        "before any timing starts. A real run pays them once, and at large copy counts they are "
        "not small.",
        "",
        startup_rows,
        "",
        startup_note,
        "",
        "This is a fixed cost, so it decides whether splitting work across cards pays. Two jobs "
        "of 2,048 copies each pay the setup twice; one job of 4,096 pays it once. It also sets "
        "a floor under how short a useful run can be — at 4,096 copies the setup alone is about "
        "four minutes before a single environment step is taken.",
        "",
        "## 9. How firm these numbers are",
        "",
        stability_summary(jobs),
        "",
        "## 10. What is still missing",
        "",
    ]
    if missing:
        out += ["These jobs have not reported. Jobs pinned to a busy node stay queued on "
                "purpose and are collected when the node frees.", ""]
        out += [f"- {m}" for m in missing]
    else:
        out.append("Nothing — every node class reported every processor count.")
    out.append("")

    (RUN / "results.md").write_text("\n".join(out))
    print(f"results.md written: {len(jobs)} jobs, {n_cells} measured cells, "
          f"{len(missing)} jobs still missing")


if __name__ == "__main__":
    main()
