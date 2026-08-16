"""A processor with fewer, faster cores against the 224-thread node, for the unified report.

Answers two questions with measurements taken on both machines under identical settings:
whether a newer, higher-clocked processor with fewer cores beats the big node per worker, and
whether it would beat it if it had the same number of hardware threads.

Exposes `sec_cpu_node_comparison()`, which returns the section's markdown, and
`fig_cpu_node_comparison()`, which writes the section's figure into the report's `figures/`.
Not wired into `make_report.py` here; it is imported from there.

Every number is read from the result files under `benchmarks/results/` and from the hardware
record the measurement jobs wrote on the nodes themselves. Nothing is typed in.
"""
import json
import re
import sys
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REPORT = HERE.parent
BASE = REPORT.parent.parent
RESULTS = BASE / "benchmarks" / "results"
FIGS = REPORT / "figures"
RUN_FOLDER = (BASE / "analysis"
              / "2026-08-15-16-39_cpu-node-comparison-newer-processor-vs-jaguar03")
HARDWARE_JSON = RUN_FOLDER / "data" / "node_hardware.json"

# the unit conversions are defined once, in the standalone report's processor module, and reused
# here so the two documents cannot come to disagree about what an hour per million steps is
sys.path.insert(0, str(BASE / "report"
                       / "2026-08-15-gpu-parallel-rl-environment-training-endtoend" / "code"))
from cpu_sections import H, K, M, style_ax                      # noqa: E402

# the node under test and the reference node, each one colour everywhere in this section
NODE_NEW = "jaguar02"
NODE_BIG = "jaguar03"
C_NEW = "#eb6834"
C_BIG = "#2a78d6"

# short runs (5 timed iterations) against long ones (150), told apart by the tag in the filename
BURST_TAGS = r"_(j2|j3)(_fill)?_p1_styleA"
SUSTAINED_TAGS = r"_(sustained|confirm)_"

# the width the rest of this report's markdown source is wrapped to
WIDTH = 95


def para(text):
    """One paragraph, wrapped to the report's line width, with the blank line after it."""
    # breaking at a hyphen would leave "last-" ending a line, and markdown rejoins lines with a
    # space, so the rendered page would read "last- level cache"
    return textwrap.fill(" ".join(text.split()), width=WIDTH, break_on_hyphens=False) + "\n\n"


def item(number, text):
    """One numbered list item, wrapped and hanging-indented under its number."""
    return textwrap.fill(" ".join(text.split()), width=WIDTH, break_on_hyphens=False,
                         initial_indent=f"{number}. ", subsequent_indent="   ") + "\n"


def load_rows(host, filename_pattern):
    """Every measured row for one host from the result files whose names match the pattern.

    The sweep was split across many invocations of the benchmark — one per worker count for the
    long runs — so one curve lives in many files and they have to be gathered before being read.
    """
    out = []
    for path in sorted(RESULTS.glob("*.json")):
        if not re.search(filename_pattern, path.name):
            continue
        record = json.loads(path.read_text())
        if (record.get("host") != host or record.get("mode") != "processes"
                or record.get("style") != "full_batch" or record.get("copies_per_proc") != 1):
            continue
        for row in record.get("rows", []):
            out.append(dict(row, _file=path.name))
    return out


def median_per_worker_count(rows):
    """One row per worker count, taking the median measurement, sorted by worker count.

    The worker counts the argument rests on were measured three or four times each. The median
    is used rather than the fastest because the question here is what a run actually gets, and
    picking the best of several repeats would bias every number upward by the spread.
    before: 224 workers measured at 0.1138, 0.1149 and 0.1131 million steps per second
    after:  one row for 224 workers carrying 0.1138
    """
    grouped = {}
    for row in rows:
        grouped.setdefault(row["workers"], []).append(row)
    out = []
    for workers, group in grouped.items():
        group.sort(key=lambda r: r["env_steps_per_sec"])
        out.append(dict(group[len(group) // 2], _repeats=len(group)))
    return sorted(out, key=lambda r: r["workers"])


def burst(host):
    """Rows from the short measurements: five timed iterations, about two seconds of work."""
    return median_per_worker_count(load_rows(host, BURST_TAGS))


def sustained(host):
    """Rows from the long measurements: 150 timed iterations, about a minute of work."""
    return median_per_worker_count(load_rows(host, SUSTAINED_TAGS))


def at_workers(rows, workers):
    """The row for one worker count, or None when that count was not measured."""
    return next((r for r in rows if r["workers"] == workers), None)


def best_config(rows):
    """The worker count that gave the machine its highest total, and that row."""
    return max(rows, key=lambda r: r["env_steps_per_sec"])


def burst_change(short_rows, long_rows, workers):
    """How far one worker's rate falls from the five-iteration measurement to the sustained one."""
    short, long = at_workers(short_rows, workers), at_workers(long_rows, workers)
    if not short or not long:
        return None
    # before: at 224 workers the short run measured 1,566 steps/s per copy, the long one 507
    # after:  {"burst": 1566, "sustained": 507, "change": -0.68}
    return {"burst": short["env_steps_per_sec_per_copy"],
            "sustained": long["env_steps_per_sec_per_copy"],
            "change": long["env_steps_per_sec_per_copy"] / short["env_steps_per_sec_per_copy"] - 1}


def largest_burst_change(short_rows, long_rows):
    """The biggest change either way, over the worker counts measured both ways."""
    shared = {r["workers"] for r in short_rows} & {r["workers"] for r in long_rows}
    changes = [burst_change(short_rows, long_rows, w)["change"] for w in shared]
    return max(changes, key=abs) if changes else None


def hardware():
    """The hardware record both measurement jobs wrote on the nodes themselves, or None."""
    if not HARDWARE_JSON.exists():
        return None
    return json.loads(HARDWARE_JSON.read_text())


def size_to_mib(size):
    """A cache size as lscpu prints it, in mebibytes. '18M' -> 18.0, '512K' -> 0.5."""
    # before: '1.3M' (lscpu -C prints one cache's size with a unit letter)
    # after:  1.3
    value, unit = float(size[:-1]), size[-1]
    return value * {"K": 1 / 1024, "M": 1.0, "G": 1024.0}[unit]


def per_core_l3_mib(facts):
    """Last-level cache in mebibytes available to each core, given how many cores share one."""
    # before: 18 MiB of level-3 cache shared by 8 cores
    # after:  2.25 MiB per core
    return size_to_mib(facts["l3_per_instance"]) / facts["cores_sharing_one_l3"]


def hardware_table():
    """The two processors side by side, on the characteristics the argument below rests on."""
    facts = hardware()
    if not facts or NODE_NEW not in facts or NODE_BIG not in facts:
        return "\n*PENDING — waiting on the hardware record from the two nodes.*\n"
    new, big = facts[NODE_NEW], facts[NODE_BIG]

    # goal: the memory-channel count, or an explicit note when the machine would not report it
    def channels(node_facts):
        block = node_facts.get("memory")
        return str(block["channels_per_socket"]) if block and block.get("channels_per_socket") \
            else "not readable"

    # goal: only characteristics a sentence below uses — a row nothing refers to is not here
    rows = [
        ("model name", new["model_name"], big["model_name"]),
        ("physical cores / hardware threads", f"{new['physical_cores']} / "
         f"{new['hardware_threads']}", f"{big['physical_cores']} / {big['hardware_threads']}"),
        ("cores / memory channels, per socket",
         f"{new['cores_per_socket']} / {channels(new)}", f"{big['cores_per_socket']} / "
         f"{channels(big)}"),
        ("measured clock, one core busy / every core busy",
         f"{new['clock_ghz_one_core_busy']:.2f} / {new['clock_ghz_every_core_busy']:.2f} GHz",
         f"{big['clock_ghz_one_core_busy']:.2f} / {big['clock_ghz_every_core_busy']:.2f} GHz"),
        ("level-3 cache per core", f"**{per_core_l3_mib(new):.2f} MiB**",
         f"**{per_core_l3_mib(big):.2f} MiB**"),
        ("one dependent access, 8 MiB working set",
         f"**{new['latency_ns']['8MiB_level3']:.1f} ns**",
         f"**{big['latency_ns']['8MiB_level3']:.1f} ns**"),
    ]
    md = (f"| characteristic | {NODE_NEW} (fewer, faster cores) | {NODE_BIG} (the big node) |\n"
          f"|---|---|---|\n")
    for name, a, b in rows:
        md += f"| {name} | {a} | {b} |\n"
    return md + "\n"


def throughput_table():
    """Both machines at the two settings the argument uses: one worker per core, one per thread."""
    facts = hardware()
    if not facts:
        return "\n*PENDING — waiting on the measurement.*\n"
    # the copies-per-worker column is 1 throughout this sweep, which is exactly why it is here:
    # without it the copies column equals the worker count and a reader cannot tell whether it
    # means copies in total or copies on each worker
    md = ("| node | worker processes | copies per worker | copies in total | seconds per "
          "iteration | million steps per second | thousand steps per second per copy | "
          "hours per million steps per copy |\n|---|---|---|---|---|---|---|---|\n")
    for host in (NODE_NEW, NODE_BIG):
        rows = sustained(host)
        for workers in (facts[host]["physical_cores"], facts[host]["hardware_threads"]):
            r = at_workers(rows, workers)
            if not r:
                continue
            md += (f"| {host} | {r['workers']} | {r['total_copies'] // r['workers']} | "
                   f"{r['total_copies']} | {r['sec_per_iteration']:.3f} | "
                   f"{M(r['env_steps_per_sec'])} | {K(r['env_steps_per_sec_per_copy'])} | "
                   f"{H(r['env_steps_per_sec_per_copy'])} |\n")
    return md + ("\n*Sustained: 150 timed iterations after 2 warm-up, median of three or four "
                 "repeats; the figure carries the full sweeps. Every number in both tables was "
                 "read on the machine it describes.*\n\n")


def second_thread_effect(host):
    """What the second hardware thread of every core does to the total, as a signed fraction."""
    facts, rows = hardware(), sustained(host)
    if not facts or host not in facts or not rows:
        return None
    cores, threads = facts[host]["physical_cores"], facts[host]["hardware_threads"]
    one_per_core, one_per_thread = at_workers(rows, cores), at_workers(rows, threads)
    if not one_per_core or not one_per_thread:
        return None
    # before: 112 workers give 0.1276 million steps/s, 224 workers give 0.1138 million
    # after:  0.1138 / 0.1276 - 1 = -0.11, that is, the second thread costs about 11 per cent
    return {"cores": cores, "threads": threads,
            "total_one_per_core": one_per_core["env_steps_per_sec"],
            "total_one_per_thread": one_per_thread["env_steps_per_sec"],
            "change": one_per_thread["env_steps_per_sec"] / one_per_core["env_steps_per_sec"] - 1,
            "share_kept": (one_per_thread["env_steps_per_sec_per_copy"]
                           / one_per_core["env_steps_per_sec_per_copy"])}


def extrapolation():
    """What the smaller processor would deliver with the big node's core and thread counts.

    Scales the smaller processor by its throughput per physical core in its own best setting,
    which is the honest unit: it already contains whatever its second hardware thread is worth,
    so the projection does not have to assume anything about that separately.
    """
    facts = hardware()
    new_rows, big_rows = sustained(NODE_NEW), sustained(NODE_BIG)
    if not (facts and new_rows and big_rows):
        return None
    new_f, big_f = facts[NODE_NEW], facts[NODE_BIG]
    new_best, big_best = best_config(new_rows), best_config(big_rows)
    big_at_threads = at_workers(big_rows, big_f["hardware_threads"])
    if not big_at_threads:
        return None
    # before: jaguar02's best setting gives 0.0227 million steps/s across its 16 physical cores
    # after:  0.0227e6 / 16 = 1,419 steps per second for each physical core
    per_core = new_best["env_steps_per_sec"] / new_f["physical_cores"]
    projected = per_core * big_f["physical_cores"]
    return {"new_best_workers": new_best["workers"],
            "new_best_total": new_best["env_steps_per_sec"],
            "new_cores": new_f["physical_cores"], "per_core": per_core,
            "big_cores": big_f["physical_cores"], "big_threads": big_f["hardware_threads"],
            "big_per_core": big_best["env_steps_per_sec"] / big_f["physical_cores"],
            "projected_total": projected,
            "big_best_workers": big_best["workers"],
            "big_best_total": big_best["env_steps_per_sec"],
            "big_total_at_threads": big_at_threads["env_steps_per_sec"],
            "share_of_big_best": projected / big_best["env_steps_per_sec"],
            "share_of_big_at_threads": projected / big_at_threads["env_steps_per_sec"]}


def work_per_cycle():
    """Per-worker rate divided by the clock it was reached at, for both machines at full load.

    The comparison that separates the two candidate explanations: a processor can be ahead
    because its clock is higher, or because it does more in each clock cycle. Dividing by the
    measured clock removes the first and leaves the second.
    """
    facts = hardware()
    if not facts:
        return None
    out = {}
    for host in (NODE_NEW, NODE_BIG):
        rows = sustained(host)
        if host not in facts or not rows:
            return None
        row = at_workers(rows, facts[host]["physical_cores"])
        if not row:
            return None
        clock = facts[host]["clock_ghz_every_core_busy"]
        # before: 1,140 steps/s per copy at a measured 2.60 GHz
        # after:  1,140 / 2.60 = 439 steps per second per copy for each gigahertz of clock
        out[host] = {"per_copy": row["env_steps_per_sec_per_copy"], "clock": clock,
                     "per_ghz": row["env_steps_per_sec_per_copy"] / clock}
    out["ratio_big_over_new"] = out[NODE_BIG]["per_ghz"] / out[NODE_NEW]["per_ghz"]
    return out


def verdict_sentence(ex):
    """The answer to the equal-thread-count question, worded to match which way the numbers went.

    Built from the two shares rather than written for one outcome, because the sustained
    measurements reversed what the short ones said and a hand-written verdict would have gone
    stale silently.
    """
    # the question as asked was the same thread count as the big node; the fairer version, each
    # machine in the setting that suits it best, is carried in the same sentence
    verb = "beats" if ex["share_of_big_at_threads"] >= 1.0 else "falls short of"
    return (f"Scaled to {ex['big_cores']} cores that is {M(ex['projected_total'])} million steps "
            f"per second, which {verb} {NODE_BIG}: {ex['share_of_big_at_threads']*100:.0f}% of "
            f"the {M(ex['big_total_at_threads'])} million it reaches at {ex['big_threads']} "
            f"threads, {ex['share_of_big_best']*100:.0f}% of its "
            f"{M(ex['big_best_total'])} million best.")


def fig_cpu_node_comparison():
    """Both nodes, aggregate rate and per-copy rate, sustained against short-burst.

    Two panels because a throughput result is two numbers that point opposite ways: the total the
    machine delivers, and what one copy gets. Both nodes are drawn on both panels, and the short
    five-iteration measurements are drawn faintly beside the sustained ones, because the gap
    between those two is itself one of the findings.
    """
    series = [(NODE_NEW, sustained(NODE_NEW), C_NEW, "o", "-", "sustained"),
              (NODE_BIG, sustained(NODE_BIG), C_BIG, "o", "-", "sustained"),
              (NODE_NEW, burst(NODE_NEW), C_NEW, "^", ":", "first seconds only"),
              (NODE_BIG, burst(NODE_BIG), C_BIG, "^", ":", "first seconds only")]
    if not any(rows for _, rows, *_ in series):
        return None

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.4), dpi=160)
    for host, rows, colour, marker, dash, kind in series:
        if not rows:
            continue
        # the faint dotted curves are the short measurements; they must not read as the result
        alpha = 1.0 if kind == "sustained" else 0.4
        x = [r["workers"] for r in rows]
        axes[0].plot(x, [r["env_steps_per_sec"] / 1e6 for r in rows], dash, color=colour,
                     linewidth=2, marker=marker, markersize=6, alpha=alpha,
                     label=f"{host}, {kind}")
        axes[1].plot(x, [r["env_steps_per_sec_per_copy"] / 1e3 for r in rows], dash, color=colour,
                     linewidth=2, marker=marker, markersize=6, alpha=alpha,
                     label=f"{host}, {kind}")

    # mark where each machine runs out of physical cores: every worker beyond that line shares a
    # core with another worker, and that is where the two machines part company
    facts = hardware() or {}
    for host, colour in ((NODE_NEW, C_NEW), (NODE_BIG, C_BIG)):
        if host not in facts:
            continue
        cores = facts[host]["physical_cores"]
        for ax in axes:
            ax.axvline(cores, color=colour, linewidth=1, linestyle="--", alpha=0.35)
        axes[1].annotate(f"{host}: {cores} cores", xy=(cores, 0.02),
                         xycoords=("data", "axes fraction"), rotation=90, fontsize=7,
                         color=colour, ha="right", va="bottom")

    axes[0].set_ylabel("million environment steps per second")
    axes[0].set_title("Total across all workers", fontsize=10)
    axes[1].set_ylabel("thousand environment steps per second per copy")
    axes[1].set_title("What one worker gets", fontsize=10)
    for ax in axes:
        ax.set_xlabel("independent worker processes, one copy each (log scale)")
        ax.set_xscale("log", base=2)
        style_ax(ax)
    axes[0].set_yscale("log")
    for ax in axes:
        ax.legend(fontsize=7.5, frameon=False)
    fig.tight_layout()
    FIGS.mkdir(exist_ok=True)
    out = FIGS / "cpu_node_comparison.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def sec_cpu_node_comparison():
    """Section: a processor with fewer, faster cores against the 224-thread node."""
    facts = hardware()
    new_rows, big_rows = sustained(NODE_NEW), sustained(NODE_BIG)
    smt_new, smt_big = second_thread_effect(NODE_NEW), second_thread_effect(NODE_BIG)
    ex, wpc = extrapolation(), work_per_cycle()
    # every part of this section is built from all of these, so a missing one is reported rather
    # than allowed to raise from inside a format string
    if not (facts and new_rows and big_rows and smt_new and smt_big and ex and wpc):
        return ("## A processor with fewer, faster cores against the 224-thread node\n\n"
                "*PENDING — waiting on the two nodes' measurements.*\n")
    fig_cpu_node_comparison()
    new, big = facts[NODE_NEW], facts[NODE_BIG]
    # the correction: the same worker counts timed for two seconds and then for a minute
    at_cores = burst_change(burst(NODE_BIG), big_rows, big["physical_cores"])
    at_threads = burst_change(burst(NODE_BIG), big_rows, big["hardware_threads"])
    new_worst = largest_burst_change(burst(NODE_NEW), new_rows)
    figure = (FIGS / "cpu_node_comparison.png").relative_to(REPORT)
    # the maker of each part, taken from the model name the table below prints in full
    # before: "Intel(R) Xeon(R) Gold 6334 CPU @ 3.60GHz" / "AMD EPYC 7663 56-Core Processor"
    # after:  "Intel" / "AMD"
    maker = {h: facts[h]["model_name"].split()[0].replace("(R)", "") for h in (NODE_NEW, NODE_BIG)}

    md = "## A processor with fewer, faster cores against the 224-thread node\n\n"
    # the two machines and the one configuration both were measured under
    md += para(
        f"{NODE_NEW}'s {maker[NODE_NEW]} part has {new['physical_cores']} fast cores against the "
        f"{big['physical_cores']} slower ones of {NODE_BIG}'s {maker[NODE_BIG]} part. Both were "
        f"held exclusively and ran this report's processor configuration: independent worker "
        f"processes, one thread and one training copy each, one update per batch.")
    # the correction to the processor numbers published earlier in this report
    md += para(
        f"**The processor numbers earlier in this report are too high.** They come from five "
        f"timed iterations, about two seconds of work; at 150 iterations {NODE_BIG} "
        f"loses {-at_cores['change']*100:.0f}% per worker at {big['physical_cores']} workers "
        f"({K(at_cores['burst'])} down to {K(at_cores['sustained'])} thousand steps per second "
        f"per copy) and {-at_threads['change']*100:.0f}% at {big['hardware_threads']} "
        f"({K(at_threads['burst'])} down to {K(at_threads['sustained'])}), while {NODE_NEW} moves "
        f"at most {abs(new_worst)*100:.0f}%. Both causes grow with the worker count: "
        f"{NODE_BIG}'s clock has not yet fallen to the "
        f"{big['clock_ghz_every_core_busy']:.2f} GHz it holds under load, and hundreds of "
        f"processes are still starting, so each measures an emptier machine than a real run "
        f"sees. Everything below is sustained.")
    md += hardware_table()
    md += throughput_table()
    md += f"![Both nodes, total and per-copy]({figure})\n\n"
    # the setting finding, and the cache reading of it that the projection below reuses
    md += para(
        f"**{NODE_BIG}'s best setting is {smt_big['cores']} workers, not {smt_big['threads']}.** "
        f"One worker on every hardware thread costs {-smt_big['change']*100:.0f}% of the total, "
        f"{M(smt_big['total_one_per_core'])} down to {M(smt_big['total_one_per_thread'])} "
        f"million steps per second: two workers sharing a core keep only "
        f"{smt_big['share_kept']*100:.0f}% each of a lone worker's rate, because what halves "
        f"between them is the last-level cache this workload depends on. On {NODE_NEW} each "
        f"keeps {smt_new['share_kept']*100:.0f}% and the total rises "
        f"{smt_new['change']*100:.0f}%, so all {smt_new['threads']} threads is its best setting.")

    md += "### If the newer processor had 224 threads\n\n"
    # the projection, stated as an upper bound rather than an estimate
    md += para(
        f"{NODE_NEW}'s best setting delivers {ex['per_core']:,.0f} steps per second per physical "
        f"core, against {ex['big_per_core']:,.0f} on {NODE_BIG}. {verdict_sentence(ex)} That is "
        f"an upper bound: it assumes a core of this design would work as fast in a "
        f"{ex['big_cores']}-core part as in a {ex['new_cores']}-core one, and {NODE_BIG} is the "
        f"evidence against that.")
    md += item(1,
               f"**The clock would not survive.** {NODE_NEW} holds "
               f"{new['clock_ghz_every_core_busy']:.2f} GHz with {new['physical_cores']} cores "
               f"working; {NODE_BIG} falls from {big['clock_ghz_one_core_busy']:.2f} idle to "
               f"{big['clock_ghz_every_core_busy']:.2f} with {big['physical_cores']}; a "
               f"{ex['big_cores']}-core {NODE_NEW} would meet the same power limit.")
    md += item(2,
               f"**Memory bandwidth per core would fall with the core count.** {NODE_NEW}'s "
               f"{(new.get('memory') or {}).get('channels_per_socket', 'unreadable number of')} "
               f"memory channels per socket feed {new['cores_per_socket']} cores, one each; "
               f"{big['cores_per_socket']} cores on that same memory would leave each "
               f"{new['cores_per_socket']/big['cores_per_socket']*100:.0f}% of it.")
    md += item(3,
               f"**The last-level cache per core would shrink the same way.** Dividing today's "
               f"block among {big['cores_per_socket']} cores per socket instead of "
               f"{new['cores_per_socket']} leaves "
               f"{per_core_l3_mib(new)*new['cores_per_socket']/big['cores_per_socket']:.2f} MiB "
               f"each against {per_core_l3_mib(new):.2f}, below what {NODE_BIG} gives its own "
               f"cores — the same halving that costs {NODE_BIG} "
               f"{-smt_big['change']*100:.0f}% at {smt_big['threads']} threads.")
    md += "\n"

    md += "### What accounts for the difference\n\n"
    # clock against work per cycle, then the one characteristic the measurements point at
    md += para(
        f"Clock speed is not the answer: with every physical core busy {NODE_NEW} runs at "
        f"{wpc[NODE_NEW]['clock']:.2f} GHz and gives one worker "
        f"{K(wpc[NODE_NEW]['per_copy'])} thousand steps per second against {NODE_BIG}'s "
        f"{K(wpc[NODE_BIG]['per_copy'])} at {wpc[NODE_BIG]['clock']:.2f} GHz — a "
        f"{(wpc[NODE_NEW]['clock']/wpc[NODE_BIG]['clock']-1)*100:.0f}% higher clock buying "
        f"{(wpc[NODE_NEW]['per_copy']/wpc[NODE_BIG]['per_copy']-1)*100:.0f}% more work, so "
        f"{NODE_BIG} does {wpc['ratio_big_over_new']:.2f} times as much of it per clock cycle. "
        f"The last-level cache is where the two part: one dependent access at an 8 MiB working "
        f"set takes {new['latency_ns']['8MiB_level3']:.1f} ns on {NODE_NEW} against "
        f"{big['latency_ns']['8MiB_level3']:.1f} on {NODE_BIG}, which also has "
        f"{per_core_l3_mib(big):.2f} MiB of that cache per core against "
        f"{per_core_l3_mib(new):.2f}. That locates the difference without proving causation "
        f"inside the training loop; proving it would need the processors' own performance "
        f"counters, which this account cannot read on these nodes.")
    # the report joins its sections with a newline, so the section ends on its last line of text
    return md.rstrip("\n") + "\n"


if __name__ == "__main__":
    print(sec_cpu_node_comparison())
