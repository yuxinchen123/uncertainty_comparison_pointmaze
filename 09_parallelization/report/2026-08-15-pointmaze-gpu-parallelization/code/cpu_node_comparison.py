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
    """The two processors side by side, one characteristic per row, each with where it came from."""
    facts = hardware()
    if not facts or NODE_NEW not in facts or NODE_BIG not in facts:
        return "\n*PENDING — waiting on the hardware record from the two nodes.*\n"
    new, big = facts[NODE_NEW], facts[NODE_BIG]

    def mem(node_facts, field, suffix=""):
        # goal: a memory field, or an explicit note when the machine would not report it
        block = node_facts.get("memory")
        return f"{block[field]}{suffix}" if block and block.get(field) else "not readable"

    rows = [
        ("model name", new["model_name"], big["model_name"], "`lscpu`"),
        ("processor family / model / stepping", new["cpu_family_model_stepping"],
         big["cpu_family_model_stepping"], "`lscpu`"),
        ("sockets", f"{new['sockets']}", f"{big['sockets']}", "`lscpu`"),
        ("physical cores (whole node)", f"{new['physical_cores']}", f"{big['physical_cores']}",
         "`lscpu`"),
        ("hardware threads per core", f"{new['threads_per_core']}", f"{big['threads_per_core']}",
         "`lscpu`"),
        ("hardware threads (whole node)", f"{new['hardware_threads']}",
         f"{big['hardware_threads']}", "`lscpu`"),
        ("clock printed in the model name",
         f"{new['nominal_clock_ghz_in_model_name']} GHz"
         if new["nominal_clock_ghz_in_model_name"] else "none printed",
         f"{big['nominal_clock_ghz_in_model_name']} GHz"
         if big["nominal_clock_ghz_in_model_name"] else "none printed",
         "`lscpu` model-name string"),
        ("**measured clock, one core busy**", f"**{new['clock_ghz_one_core_busy']:.2f} GHz**",
         f"**{big['clock_ghz_one_core_busy']:.2f} GHz**",
         "dependent addition chain, run on the node"),
        ("**measured clock, every core busy**", f"**{new['clock_ghz_every_core_busy']:.2f} GHz**",
         f"**{big['clock_ghz_every_core_busy']:.2f} GHz**",
         "same chain, run while the training workload occupied every other core"),
        ("level-1 data cache per core", new["l1d_per_core"], big["l1d_per_core"], "`lscpu -C`"),
        ("level-2 cache per core", new["l2_per_core"], big["l2_per_core"], "`lscpu -C`"),
        ("level-3 cache, one block", new["l3_per_instance"], big["l3_per_instance"], "`lscpu -C`"),
        ("level-3 cache, whole node", new["l3_total"], big["l3_total"], "`lscpu -C`"),
        ("cores sharing one level-3 block", f"{new['cores_sharing_one_l3']:.0f}",
         f"{big['cores_sharing_one_l3']:.0f}", "cores divided by number of level-3 blocks"),
        ("**level-3 cache per core**", f"**{per_core_l3_mib(new):.2f} MiB**",
         f"**{per_core_l3_mib(big):.2f} MiB**", "the two rows above"),
        ("memory type", mem(new, "memory_type"), mem(big, "memory_type"),
         "EDAC labels in `/sys`"),
        ("**memory channels per socket**", f"**{mem(new, 'channels_per_socket')}**",
         f"**{mem(big, 'channels_per_socket')}**", "EDAC labels in `/sys`"),
        ("memory installed", mem(new, "dimms_total", " DIMMs"), mem(big, "dimms_total", " DIMMs"),
         "EDAC labels in `/sys`"),
        ("memory clock", "not readable", "not readable",
         "`dmidecode` needs privileges this account does not have on these nodes"),
        ("memory regions the node reports", f"{new['numa_nodes']}", f"{big['numa_nodes']}",
         "`lscpu`"),
    ]
    for name, key in [("one dependent access, 24 KiB working set", "24KiB_level1"),
                      ("one dependent access, 384 KiB working set", "384KiB_level2"),
                      ("**one dependent access, 8 MiB working set**", "8MiB_level3"),
                      ("one dependent access, 256 MiB working set", "256MiB_main_memory")]:
        bold = "**" if name.startswith("**") else ""
        rows.append((name, f"{bold}{new['latency_ns'][key]:.1f} ns{bold}",
                     f"{bold}{big['latency_ns'][key]:.1f} ns{bold}",
                     "random walk through a buffer, run on the node"))
    for name, key in [("one core reading an 8 MiB working set", "8MiB_level3"),
                      ("one core reading main memory", "256MiB_main_memory")]:
        rows.append((name, f"{new['read_bandwidth_gb_per_s_one_core'][key]:.1f} GB/s",
                     f"{big['read_bandwidth_gb_per_s_one_core'][key]:.1f} GB/s",
                     "streaming read, run on the node"))
    rows.append(("vector instructions the array library chose",
                 new["vector_instruction_set_used_by_torch"],
                 big["vector_instruction_set_used_by_torch"],
                 "`torch.backends.cpu.get_cpu_capability()`"))

    md = (f"| characteristic | {NODE_NEW} (fewer, faster cores) | {NODE_BIG} (the big node) | "
          f"where the number came from |\n|---|---|---|---|\n")
    for name, a, b, source in rows:
        md += f"| {name} | {a} | {b} | {source} |\n"
    md += ("\n*Everything above was read on the machine it describes, by the measurement job "
           "itself, not from a specification sheet. The memory clock needs privileges this "
           "account does not have on these nodes, so it is left as not readable rather than "
           "guessed. The microarchitecture names and their release years are not on the machine "
           "at all and so are not in this table either.*\n\n")
    return md


def throughput_table(rows, note):
    """A throughput table in the project's standard columns, one row per worker count."""
    if not rows:
        return "\n*PENDING — waiting on the measurement.*\n"
    # the copies-per-worker column is 1 throughout this sweep, which is exactly why it is here:
    # without it the copies column equals the worker count and a reader cannot tell whether it
    # means copies in total or copies on each worker
    md = ("| worker processes | copies per worker | copies in total | seconds per iteration | "
          "million steps per second | thousand steps per second per copy | "
          "hours per million steps per copy |\n|---|---|---|---|---|---|---|\n")
    for r in rows:
        md += (f"| {r['workers']} | {r['total_copies'] // r['workers']} | {r['total_copies']} | "
               f"{r['sec_per_iteration']:.3f} | "
               f"{M(r['env_steps_per_sec'])} | {K(r['env_steps_per_sec_per_copy'])} | "
               f"{H(r['env_steps_per_sec_per_copy'])} |\n")
    return md + f"\n*{note}*\n\n"


def full_load_table():
    """The two machines compared at equal load: every physical core of each running one worker."""
    facts, new_rows, big_rows = hardware(), sustained(NODE_NEW), sustained(NODE_BIG)
    if not (facts and new_rows and big_rows):
        return "\n*PENDING — waiting on the sustained measurements.*\n"
    md = ("| node | workers (one per physical core) | measured clock while running | "
          "thousand steps per second per copy | hours per million steps per copy | "
          "steps per second per copy for each gigahertz of clock |\n|---|---|---|---|---|---|\n")
    for host in (NODE_NEW, NODE_BIG):
        rows, f = sustained(host), facts[host]
        row = at_workers(rows, f["physical_cores"])
        if not row:
            continue
        # dividing the rate by the clock it was reached at separates "runs faster" from
        # "does more in each cycle", which is the whole question in the explanation below
        per_ghz = row["env_steps_per_sec_per_copy"] / f["clock_ghz_every_core_busy"]
        md += (f"| {host} | {f['physical_cores']} | {f['clock_ghz_every_core_busy']:.2f} GHz | "
               f"{K(row['env_steps_per_sec_per_copy'])} | "
               f"{H(row['env_steps_per_sec_per_copy'])} | {per_ghz:.0f} |\n")
    return md + ("\n*This is the comparison at equal load: each machine has every one of its "
                 "physical cores running one worker.*\n\n")


def second_thread_effect(host):
    """What the second hardware thread of every core does to the total, as a signed fraction."""
    facts, rows = hardware(), sustained(host)
    if not facts or host not in facts or not rows:
        return None
    cores, threads = facts[host]["physical_cores"], facts[host]["hardware_threads"]
    one_per_core, one_per_thread = at_workers(rows, cores), at_workers(rows, threads)
    if not one_per_core or not one_per_thread:
        return None
    # before: 112 workers give 0.1296 million steps/s, 224 workers give 0.1138 million
    # after:  0.1138 / 0.1296 - 1 = -0.12, that is, the second thread costs about 12 per cent
    return {"cores": cores, "threads": threads,
            "total_one_per_core": one_per_core["env_steps_per_sec"],
            "total_one_per_thread": one_per_thread["env_steps_per_sec"],
            "change": one_per_thread["env_steps_per_sec"] / one_per_core["env_steps_per_sec"] - 1,
            "per_copy_one_per_core": one_per_core["env_steps_per_sec_per_copy"],
            "per_copy_one_per_thread": one_per_thread["env_steps_per_sec_per_copy"]}


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
    # before: jaguar02's best setting gives 0.0230 million steps/s across its 16 physical cores
    # after:  0.0230e6 / 16 = 1,437 steps per second for each physical core
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
        # before: 1,280 steps/s per copy at a measured 3.56 GHz
        # after:  1,280 / 3.56 = 359 steps per second per copy for each gigahertz of clock
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
    # the question as asked: the same thread count as the big node
    if ex["share_of_big_at_threads"] >= 1.0:
        head = (f"**The answer is yes.** Scaled to {ex['big_cores']} cores and "
                f"{ex['big_threads']} hardware threads, {NODE_NEW} projects to "
                f"{M(ex['projected_total'])} million environment steps per second, against "
                f"{M(ex['big_total_at_threads'])} million measured on {NODE_BIG} at that same "
                f"thread count — {ex['share_of_big_at_threads']*100:.0f}% of it.")
    else:
        head = (f"**The answer is no.** Scaled to {ex['big_cores']} cores and "
                f"{ex['big_threads']} hardware threads, {NODE_NEW} projects to "
                f"{M(ex['projected_total'])} million environment steps per second against "
                f"{M(ex['big_total_at_threads'])} million measured on {NODE_BIG} — "
                f"{ex['share_of_big_at_threads']*100:.0f}% of it.")
    # the fairer version of the same question: each machine in the setting that suits it best
    fair = (f"Comparing each machine in the setting that suits it best is fairer to {NODE_BIG}, "
            f"whose best is {ex['big_best_workers']} workers at "
            f"{M(ex['big_best_total'])} million rather than {ex['big_threads']} workers, and the "
            f"projection still comes out ahead: {ex['share_of_big_best']*100:.0f}% of "
            f"{NODE_BIG}'s best.")
    return f"{head} {fair}"


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
    new_best = best_config(new_rows)

    md = f"""## A processor with fewer, faster cores against the 224-thread node

### The question

The node measured everywhere else in this report, {NODE_BIG}, wins by having a great many cores.
The opposite kind of machine also exists on this cluster: fewer cores, each clocked higher. Since
this workload is one single-thread process per worker running a small policy network and a
physics environment — many tiny dependent operations rather than large parallel arithmetic — the
expectation going in was that the higher-clocked processor would give each worker more steps per
second, and that the big node would win only on the total.

{NODE_NEW} was chosen to test that. It carries the highest clock printed on any processor in this
cluster and the fewest cores per socket of its generation, and it was completely idle, so it
could be held exclusively for the measurement exactly as {NODE_BIG} was. The genuinely newer
processors here, the Zen 4 parts in the serval machines, could not be used: one sits inside a
maintenance reservation until the end of the month and the other four were each carrying another
user's multi-day job, so none could be held exclusively.

### The two processors side by side

{hardware_table()}
### What was measured, and why the measurement had to be made twice

Independent worker processes, one thread each, one training copy each, one update per batch — the
same configuration on both machines, and the same configuration the rest of this report's
processor numbers use. The worker count runs from one up to each node's physical core count and
then to its hardware thread count.

Each point was measured with five timed iterations, about two seconds of work, and then again
with 150 timed iterations, about a minute. That turned out to matter more than anything else in
the setup. Two separate effects make a two-second measurement read high, and both of them grow
with the worker count:

1. **The clock has not settled.** A server processor runs above its sustained clock for the first
   seconds of a load. With every core busy {NODE_NEW} holds
   {new['clock_ghz_every_core_busy']:.2f} GHz, essentially its idle
   {new['clock_ghz_one_core_busy']:.2f} GHz, while {NODE_BIG} settles from
   {big['clock_ghz_one_core_busy']:.2f} GHz down to {big['clock_ghz_every_core_busy']:.2f} GHz.
   A machine with 16 cores can hold its clock with all of them working; one with 112 cannot.
2. **The workers do not overlap.** The benchmark adds up each worker's own rate. Over seven
   iterations, several hundred processes spend much of that time starting up at different
   moments, so each one measures a machine that is emptier than the machine a real run sees, and
   the sum describes a load that never existed. Over 150 iterations they all run together
   throughout and the sum is honest.

The result is that the short numbers are wrong in a way that gets worse exactly where the
argument is decided:

| node | workers | first seconds only, thousand steps/s per copy | sustained, thousand steps/s per copy | difference |
|---|---|---|---|---|
"""
    # the burst-against-sustained comparison, at the worker counts measured both ways
    for host in (NODE_NEW, NODE_BIG):
        b, s = burst(host), sustained(host)
        for w in sorted({r["workers"] for r in b} & {r["workers"] for r in s}):
            rb, rs = at_workers(b, w), at_workers(s, w)
            drop = rs["env_steps_per_sec_per_copy"] / rb["env_steps_per_sec_per_copy"] - 1
            md += (f"| {host} | {w} | {K(rb['env_steps_per_sec_per_copy'])} | "
                   f"{K(rs['env_steps_per_sec_per_copy'])} | {drop*100:+.0f}% |\n")

    md += f"""
Everything from here on uses the sustained numbers, and every worker count the argument rests on
was measured three or four times. The short measurements are kept in the figure, drawn faintly,
because the earlier processor sections of this report were built from five-iteration runs and
their figures for {NODE_BIG} at high worker counts are therefore too high.

### The measured comparison

{full_load_table()}![Both nodes, total and per-copy]({(FIGS / 'cpu_node_comparison.png').relative_to(REPORT)})

Full sweeps, sustained:

{throughput_table(new_rows, f'{NODE_NEW}: {new["physical_cores"]} physical cores, '
                            f'{new["hardware_threads"]} hardware threads. Independent worker '
                            f'processes, one copy each, one update per batch, 150 timed '
                            f'iterations after 2 warm-up iterations.')}\
{throughput_table(big_rows, f'{NODE_BIG}: {big["physical_cores"]} physical cores, '
                            f'{big["hardware_threads"]} hardware threads. Same configuration.')}\
### The second hardware thread per core

Every core here can run two workers at once. Whether that is worth doing is where the two
machines differ most, and it is the opposite of what the short measurements suggested.

| node | one worker per physical core | one worker per hardware thread | what the second thread does to the total |
|---|---|---|---|
| {NODE_NEW} | {smt_new['cores']} workers, {M(smt_new['total_one_per_core'])} million steps/s | {smt_new['threads']} workers, {M(smt_new['total_one_per_thread'])} million steps/s | **{smt_new['change']*100:+.0f}%** |
| {NODE_BIG} | {smt_big['cores']} workers, {M(smt_big['total_one_per_core'])} million steps/s | {smt_big['threads']} workers, {M(smt_big['total_one_per_thread'])} million steps/s | **{smt_big['change']*100:+.0f}%** |

The arithmetic is simple: put two workers on a core and each of them keeps some fraction of what
a lone worker on that core was getting. If that fraction is above one half the pair is worth it,
and below one half it is not. On {NODE_NEW} each of the two keeps
{smt_new['per_copy_one_per_thread']/smt_new['per_copy_one_per_core']*100:.0f}% — above half, so
the total rises {smt_new['change']*100:+.0f}%. On {NODE_BIG} each keeps only
{smt_big['per_copy_one_per_thread']/smt_big['per_copy_one_per_core']*100:.0f}% — below half, so
the total falls {smt_big['change']*100:+.0f}%. **Filling all {smt_big['threads']} of
{NODE_BIG}'s hardware threads is worse than using only its {smt_big['cores']} physical cores.**
That single fact decides the extrapolation below, and it is invisible in a two-second
measurement, which reported the second thread as nearly doubling {NODE_BIG}'s total.

### If the newer processor had 224 threads

Take {NODE_NEW} in its best setting — {ex['new_best_workers']} workers,
{M(ex['new_best_total'])} million environment steps per second across its {ex['new_cores']}
physical cores, that is {ex['per_core']:,.0f} steps per second for each physical core — and
assume that per-core figure would survive if the same processor design had {ex['big_cores']}
cores instead of {ex['new_cores']}. For comparison {NODE_BIG} delivers
{ex['big_per_core']:,.0f} steps per second per physical core in its own best setting.

| what is being counted | projected {NODE_NEW} at {ex['big_cores']} cores | measured {NODE_BIG} | share |
|---|---|---|---|
| against {NODE_BIG} at {ex['big_threads']} hardware threads | {M(ex['projected_total'])} million steps/s | {M(ex['big_total_at_threads'])} million steps/s | **{ex['share_of_big_at_threads']*100:.0f}%** |
| against {NODE_BIG} at its best ({ex['big_best_workers']} workers) | {M(ex['projected_total'])} million steps/s | {M(ex['big_best_total'])} million steps/s | **{ex['share_of_big_best']*100:.0f}%** |

{verdict_sentence(ex)}

**The assumption this rests on, stated plainly:** that a core in a {ex['new_cores']}-core version
of this processor and a core in a {ex['big_cores']}-core version of it would do the same amount
of work per second. Everything that is known about these two machines says that assumption is
optimistic, and {NODE_BIG} is itself the evidence, because {NODE_BIG} *is* the experiment of
putting 112 cores behind one memory system:

1. **The clock would not survive.** Measured, not supposed. {NODE_NEW} holds
   {new['clock_ghz_every_core_busy']:.2f} GHz with {new['physical_cores']} cores working;
   {NODE_BIG} falls to {big['clock_ghz_every_core_busy']:.2f} GHz with {big['physical_cores']}
   working, from {big['clock_ghz_one_core_busy']:.2f} GHz idle. A {ex['big_cores']}-core version
   of {NODE_NEW} would meet the same power budget and give up clock the same way.
2. **Memory bandwidth per core would fall sevenfold.** {NODE_NEW} has
   {(new.get('memory') or {}).get('channels_per_socket', 'an unreadable number of')} memory
   channels feeding {new['cores_per_socket']} cores on each socket, which is one channel per
   core. Growing that socket to {big['cores_per_socket']} cores without widening its memory
   would leave each core a seventh of the bandwidth the measured part enjoys. ({NODE_BIG}'s own
   channel count cannot be read: it exposes no memory-controller entries in `/sys`, so this
   point is made from {NODE_NEW}'s readable figure alone rather than from a comparison.)
3. **The shared cache per core would shrink the same way** — and this one is already visible in
   the measurements. {NODE_BIG}'s total *falls* when it goes from {smt_big['cores']} workers to
   {smt_big['threads']} precisely because each worker's share of the level-3 cache halves. A
   {ex['big_cores']}-core {NODE_NEW} keeping today's {new['l3_per_instance']} block per socket
   would be dividing it among {big['cores_per_socket']} cores instead of
   {new['cores_per_socket']}: {per_core_l3_mib(new)*new['cores_per_socket']/big['cores_per_socket']:.2f} MiB
   per core rather than {per_core_l3_mib(new):.2f}, which is less than {NODE_BIG} gives each of
   its own cores.

So the projection is an upper bound, not an estimate. The honest summary is that the two designs
are closer than their core counts suggest, that {NODE_NEW}'s cores are worth more each, and that
scaling {NODE_NEW} up to {ex['big_cores']} cores would run into the very effects that are
currently holding {NODE_BIG} back. The one asymmetry to keep in mind: only one side of this table
is a projection. {NODE_BIG}'s numbers are measured on real workers on a real machine.

### What accounts for the difference

The starting expectation was that clock speed would decide this. The measurement says clock speed
is real but secondary, and it says so twice.

**Per worker, at equal load, the higher-clocked machine wins — by much less than its clock.**
With every physical core busy, {NODE_NEW} runs at {wpc[NODE_NEW]['clock']:.2f} GHz and gives one
worker {K(wpc[NODE_NEW]['per_copy'])} thousand steps per second; {NODE_BIG} runs at
{wpc[NODE_BIG]['clock']:.2f} GHz and gives one worker {K(wpc[NODE_BIG]['per_copy'])} thousand.
{NODE_NEW}'s clock is {(wpc[NODE_NEW]['clock']/wpc[NODE_BIG]['clock']-1)*100:.0f}% higher but its
per-worker rate is only {(wpc[NODE_NEW]['per_copy']/wpc[NODE_BIG]['per_copy']-1)*100:.0f}% higher.
Divide each rate by the clock it was reached at and {NODE_BIG} does
**{wpc['ratio_big_over_new']:.2f} times** as much of this work in every clock cycle. Its cores are
individually the more effective ones at this workload; it gives most of that back by running them
slower, which is the price of having 112 of them.

**Which characteristic produces that?** The evidence points at the last-level cache — not the
clock, not the vector units, not main memory:

1. **The two are close in the small caches, and {NODE_NEW} is ahead in main memory.** One
   dependent access at a 24 KiB working set takes {new['latency_ns']['24KiB_level1']:.1f} ns on
   {NODE_NEW} against {big['latency_ns']['24KiB_level1']:.1f} on {NODE_BIG}; at 384 KiB,
   {new['latency_ns']['384KiB_level2']:.1f} against {big['latency_ns']['384KiB_level2']:.1f}; in
   main memory {NODE_NEW} is the faster of the two,
   {new['latency_ns']['256MiB_main_memory']:.1f} ns against
   {big['latency_ns']['256MiB_main_memory']:.1f}. {NODE_NEW} also has more level-2 cache per core
   ({new['l2_per_core']} against {big['l2_per_core']}). None of that explains a per-cycle deficit.
2. **At the last-level cache the order reverses, by a wide margin.** At an 8 MiB working set one
   dependent access takes {new['latency_ns']['8MiB_level3']:.1f} ns on {NODE_NEW} and
   {big['latency_ns']['8MiB_level3']:.1f} ns on {NODE_BIG} — {NODE_BIG} reaches data of that size
   **{new['latency_ns']['8MiB_level3']/big['latency_ns']['8MiB_level3']:.1f} times faster**. One
   core streaming the same working set reads
   {new['read_bandwidth_gb_per_s_one_core']['8MiB_level3']:.0f} GB/s on {NODE_NEW} against
   {big['read_bandwidth_gb_per_s_one_core']['8MiB_level3']:.0f} GB/s on {NODE_BIG}. And
   {NODE_BIG} has {per_core_l3_mib(big)/per_core_l3_mib(new):.1f} times as much of that cache per
   core ({per_core_l3_mib(big):.2f} MiB against {per_core_l3_mib(new):.2f} MiB), because its
   cache is cut into small blocks shared by {big['cores_sharing_one_l3']:.0f} cores each rather
   than one large block shared by {new['cores_sharing_one_l3']:.0f}.
3. **The vector units are not the answer, and the measurement shows it.** The array library
   compiled itself for {new['vector_instruction_set_used_by_torch']} on {NODE_NEW} and only
   {big['vector_instruction_set_used_by_torch']} on {NODE_BIG} — the wider, more modern vector
   instructions are on the machine that does *less* work per clock cycle. That is what a workload
   of many small dependent operations looks like: the wide arithmetic has little to do, and the
   processor that keeps a few megabytes close to the core wins instead.

The same reading explains why {NODE_BIG} loses throughput when its second hardware threads are
used. Two workers on one core share that core's caches. {NODE_NEW} has
{new['l2_per_core']} of level-2 cache per core to divide between the two, which still leaves each
of them more than a {NODE_BIG} core has to give a single worker ({big['l2_per_core']}), so on
{NODE_NEW} the second worker is worth having. On {NODE_BIG} the share being halved is the one the
workload already depends on, and halving it costs more than the second worker brings.

**One honest limit.** These measurements locate the difference — at the last-level cache rather
than at the clock, the vector width, or main memory — by timing the memory system at each
working-set size on both machines. They do not prove causation inside the training loop itself;
that would need the processors' own performance counters during the run, which this account
cannot read on these nodes. What can be said without qualification is what was measured directly:
the higher-clocked processor wins per worker by far less than its clock advantage, it does less
work per clock cycle, filling {NODE_BIG}'s second hardware threads makes {NODE_BIG} slower rather
than faster, and a {ex['big_threads']}-thread version of {NODE_NEW} projects
{ex['share_of_big_best']*100:.0f}% of {NODE_BIG}'s best — under an assumption that the evidence
says is generous.
"""
    return md


if __name__ == "__main__":
    print(sec_cpu_node_comparison())
