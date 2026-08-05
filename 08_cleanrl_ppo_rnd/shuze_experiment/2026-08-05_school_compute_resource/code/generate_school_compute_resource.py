"""Render the school compute-resource document from the node inventory JSON.

Reads ../data/cluster_nodes.json (written by build_cluster_nodes_json.py from live Slurm plus the
shared GPU catalog) and writes ../2026-08-05_school_compute_resource.md: one table per partition
(cpu, gpu, nolim, gnolim) with nodes of identical hardware folded into a single row, a table adding
the four partitions up, the per-user limit table, and the GPU-generation compatibility section.
Every table of countable things ends in a total row.

Run:
    PYTHONNOUSERSITE=1 /p/rlprojects/RND/.venvs/exploration/bin/python \
        generate_school_compute_resource.py
"""

import json
import os
import re

# The four partitions, in the order sinfo lists them. Each entry is (partition name, one-line
# description used as the table caption).
PARTITIONS = [
    ("cpu", "no GPUs; the largest core pool on the cluster"),
    ("gpu", "every GPU node that is not in gnolim; a 4-day time limit"),
    ("nolim", "no GPUs, a 20-day time limit, small core pool"),
    ("gnolim", "older GPU nodes with a 20-day time limit"),
]

# Column headers of the per-partition node tables. Kept literal: each one names exactly what the
# cell holds. "N/A" fills the GPU columns of a node that has no GPU. The hardware columns hold
# per-node figures, so a row that folds several identical nodes still reads as one machine; the
# <br> in two headers keeps those columns narrow enough to print.
NODE_COLUMNS = [
    "node",
    "number<br>of nodes",
    "state",
    "cpu model",
    "sockets x cores x threads",
    "cpus counted by slurm<br>(per node)",
    "system memory<br>(per node)",
    "gpus per node",
    "gpu model",
    "gpu memory per card",
    "gpu architecture (year)",
    "gpu compute capability",
]

# States in which a node is simply running work. A node in any other state (down, drained, reserved)
# is kept on a row of its own, so its name and its state stay visible instead of disappearing into a
# folded range.
PLAIN_STATES = {"idle", "mixed", "allocated"}


def format_memory_gib(mb):
    """Turn a memory figure in MB into a short GiB string, or 'N/A' when the field is absent."""
    if mb is None:
        return "N/A"
    return f"{mb / 1024:.0f} GiB"


def format_gpu_memory(node):
    """Render a node's per-card GPU memory, marking values the catalog flags as approximate."""
    # A node with no GPU has every GPU field set to None.
    if node.get("gpu_count", 0) == 0:
        return "N/A"
    gib = node.get("gpu_memory_gib_per_card")
    if gib is None:
        return "unknown"
    # The catalog marks a value "approximate" when it was not read off the card itself. Those are
    # shown with a trailing asterisk, the same convention the previous resource document used.
    mark = "*" if node.get("gpu_memory_label") == "approximate" else ""
    return f"{gib:.0f} GiB{mark}"


# Values read off the cards themselves, which override the shared catalog where the two disagree.
# The catalog is the input to the group's GPU packing tools, so a wrong entry there decides which
# nodes a sweep is allowed to use — titanx03 was excluded from a sweep by its wrong entry.
MEASURED_OVERRIDES = {
    "titanx03": {
        "gpu_model": "TITAN X (Pascal)",
        "gpu_architecture": "Pascal",
        "gpu_architecture_year": 2016,
        "gpu_compute_capability": 6.1,
        "gpu_memory_gib_per_card": 11.9,
        "gpu_memory_label": "measured",
        "source": "nvidia-smi and torch.cuda.get_device_properties on the node, Slurm job 6533825, "
                  "2026-08-05. The catalog records a Maxwell Titan X at compute capability 5.2.",
    },
}


def apply_measured_overrides(nodes):
    """Replace catalog values with values read off the card, returning the list of corrections."""
    corrections = []
    for n in nodes:
        override = MEASURED_OVERRIDES.get(n["node"])
        if not override:
            continue
        for field, new in override.items():
            if field == "source":
                continue
            old = n.get(field)
            if old != new:
                corrections.append((n["node"], field, old, new))
                n[field] = new
    return corrections


def format_architecture(node):
    """Render 'Family (year)' for a GPU node, 'N/A' for a node with no GPU."""
    if node.get("gpu_count", 0) == 0:
        return "N/A"
    fam = node.get("gpu_architecture") or "unknown"
    year = node.get("gpu_architecture_year")
    return f"{fam} ({year})" if year else fam


def split_node_name(name):
    """Split a node name into (letter prefix, trailing number or None, digit width)."""
    # before: "affogato07" / "slurm1" / "hydro"
    # after:  ("affogato", 7, 2) / ("slurm", 1, 1) / ("hydro", None, 0)
    m = re.fullmatch(r"([A-Za-z_]+)([0-9]*)", name)
    if not m:
        raise ValueError(f"node name {name!r} is not letters followed by digits")
    prefix, digits = m.group(1), m.group(2)
    return prefix, (int(digits) if digits else None), len(digits)


def format_node_range(names):
    """Compress node names sharing a prefix into Slurm hostlist form, e.g. 'cortado[01-05,07-10]'."""
    # Slurm's own --nodelist syntax, so a row can be pasted straight into an sbatch line.
    # before: ["cortado01","cortado02","cortado03","cortado04","cortado05","cortado07", ... ,"cortado10"]
    # after:  "cortado[01-05,07-10]"   (a single name stays itself: ["hydro"] -> "hydro")
    if len(names) == 1:
        return names[0]
    parts = [split_node_name(n) for n in names]
    prefix = parts[0][0]
    width = parts[0][2]
    if any(p != prefix for p, _, _ in parts) or any(w != width for _, _, w in parts):
        raise ValueError(f"cannot fold names with mixed prefix or digit width: {names}")
    # Walk the sorted numbers, closing a run whenever the next number is not one higher.
    numbers = sorted(n for _, n, _ in parts)
    runs, start, prev = [], numbers[0], numbers[0]
    for n in numbers[1:]:
        if n != prev + 1:
            runs.append((start, prev))
            start = n
        prev = n
    runs.append((start, prev))
    pieces = [f"{a:0{width}d}" if a == b else f"{a:0{width}d}-{b:0{width}d}" for a, b in runs]
    return f"{prefix}[{','.join(pieces)}]"


def fold_key(node):
    """Key that decides which nodes share a row: name prefix plus every hardware field shown."""
    state = node.get("state", "").lower()
    return (
        split_node_name(node["node"])[0],
        node.get("cpu_model"),
        node.get("sockets"),
        node.get("cores_per_socket"),
        node.get("threads_per_core"),
        node.get("cpus_total_slurm"),
        node.get("real_memory_mb"),
        node.get("gpu_count", 0),
        node.get("gpu_model"),
        format_gpu_memory(node),
        format_architecture(node),
        node.get("gpu_compute_capability"),
        # A node that is not simply running work keeps its own row, keyed by its own name.
        None if state in PLAIN_STATES else node["node"],
    )


def fold_nodes(part_nodes):
    """Group a partition's nodes into rows of identical hardware, sorted by first node name."""
    # before: 10 cortado dicts, 6 of them mixed, 3 idle, 1 down+not_responding
    # after:  [[cortado01..05,07..10 (9 dicts)], [cortado06 (1 dict)]]
    groups = {}
    for n in part_nodes:
        groups.setdefault(fold_key(n), []).append(n)
    ordered = [sorted(g, key=lambda n: n["node"]) for g in groups.values()]
    return sorted(ordered, key=lambda g: split_node_name(g[0]["node"])[:2])


def format_states(group):
    """Render the states of a folded group as counts, e.g. '6 mixed, 3 idle'; one node keeps its state."""
    states = [n.get("state", "").lower() for n in group]
    if len(group) == 1:
        return states[0]
    counts = {s: states.count(s) for s in set(states)}
    return ", ".join(f"{counts[s]} {s}" for s in sorted(counts, key=lambda s: (-counts[s], s)))


def node_row(group):
    """Build one markdown table row for a folded group of identical nodes, in NODE_COLUMNS order."""
    # Sockets/cores/threads come straight from scontrol; they multiply out to "cpus counted by
    # slurm", which counts hardware threads, not cores. Every hardware figure below is per node,
    # so a folded row reads exactly like a single-node row.
    # before: sockets=2, cores_per_socket=8, threads_per_core=2
    # after:  "2 x 8 x 2"  and  cpus counted by slurm = 32
    node = group[0]
    geometry = "{} x {} x {}".format(
        node.get("sockets"), node.get("cores_per_socket"), node.get("threads_per_core")
    )
    has_gpu = node.get("gpu_count", 0) > 0
    return [
        f"`{format_node_range([n['node'] for n in group])}`",
        str(len(group)),
        format_states(group),
        node.get("cpu_model") or "unknown",
        geometry,
        str(node.get("cpus_total_slurm", "")),
        format_memory_gib(node.get("real_memory_mb")),
        str(node["gpu_count"]) if has_gpu else "N/A",
        node.get("gpu_model") or "N/A",
        format_gpu_memory(node),
        format_architecture(node),
        f"{node['gpu_compute_capability']:.1f}" if has_gpu and node.get("gpu_compute_capability") else "N/A",
    ]


def node_total_row(part_nodes):
    """Build the closing total row of a node table: node count, cpus, memory and gpus added up."""
    # The three summed cells say "(in total)" because their column headers read "per node".
    cpus = sum(n["cpus_total_slurm"] for n in part_nodes)
    gpus = sum(n["gpu_count"] for n in part_nodes)
    mem_tib = sum(n["real_memory_mb"] for n in part_nodes) / 1024 / 1024
    return [
        "**total**",
        f"**{len(part_nodes)}**",
        "N/A",
        "N/A",
        "N/A",
        f"**{cpus}**<br>(in total)",
        f"**{mem_tib:.1f} TiB**<br>(in total)",
        f"**{gpus}**<br>(in total)",
        "N/A",
        "N/A",
        "N/A",
        "N/A",
    ]


def render_table(headers, rows):
    """Render a markdown table from a header list and a list of row-cell lists."""
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def render_partition_totals_table(partitions_meta, nodes, totals):
    """Render the table that adds the four partitions up, ending in the whole-cluster total row."""
    # Every node sits in exactly one partition, so the four partition rows add up to the cluster
    # figures reported by the inventory; disagreement means the inventory changed shape.
    headers = ["partition", "number of nodes", "cpus counted by slurm", "gpus", "system memory"]
    rows = []
    for meta in partitions_meta:
        part_nodes = [n for n in nodes if meta["partition"] in n["partitions"]]
        rows.append(
            [
                f"`{meta['partition']}`",
                str(len(part_nodes)),
                str(sum(n["cpus_total_slurm"] for n in part_nodes)),
                str(sum(n["gpu_count"] for n in part_nodes)),
                f"{sum(n['real_memory_mb'] for n in part_nodes) / 1024 / 1024:.1f} TiB",
            ]
        )
    summed = (
        sum(int(r[1]) for r in rows),
        sum(int(r[2]) for r in rows),
        sum(int(r[3]) for r in rows),
    )
    expected = (totals["node_count"], totals["cpus_total_slurm"], totals["gpu_count"])
    if summed != expected:
        raise ValueError(f"partition rows add up to {summed}, inventory totals say {expected}")
    rows.append(
        [
            "**all four partitions**",
            f"**{summed[0]}**",
            f"**{summed[1]}**",
            f"**{summed[2]}**",
            f"**{totals['real_memory_mb'] / 1024 / 1024:.1f} TiB**",
        ]
    )
    return render_table(headers, rows)


def partition_summary(part_meta, part_nodes):
    """Build the one-paragraph summary line that sits above a partition's node table."""
    # Sum the per-node figures rather than trusting a partition-level total, so the paragraph and
    # the table below it can never disagree.
    gpus = sum(n["gpu_count"] for n in part_nodes)
    cpus = sum(n["cpus_total_slurm"] for n in part_nodes)
    mem_tib = sum(n["real_memory_mb"] for n in part_nodes) / 1024 / 1024
    limits = part_meta["per_user_limits_live"]
    gpu_txt = f", {gpus} GPUs" if gpus else ", no GPUs"
    return (
        f"{len(part_nodes)} nodes, {cpus} cpus counted by slurm{gpu_txt}, "
        f"{mem_tib:.1f} TiB of system memory in total. "
        f"Time limit {part_meta['max_time']}, qos `{part_meta['qos']}`. "
        f"Per user: {limits['cpu']} cpus"
        + (f", {limits['gpu']} gpus" if limits.get("gpu") else ", no gpu cap (no gpus here)")
        + f", {limits['mem_mb'] / 1024 / 1024:.0f} TiB memory."
    )


def render_limits_table(partitions, reservation_qos):
    """Render the per-partition per-user limit table, with the reservation qos as a final row."""
    headers = [
        "partition",
        "qos",
        "time limit",
        "gpus in partition",
        "per-user cpu cap",
        "per-user gpu cap",
        "per-user memory cap",
    ]
    rows = []
    for meta in partitions:
        lim = meta["per_user_limits_live"]
        rows.append(
            [
                f"`{meta['partition']}`",
                f"`{meta['qos']}`",
                meta["max_time"],
                str(meta["gpu_count_live"]) if meta["gpu_count_live"] else "N/A",
                str(lim["cpu"]),
                str(lim["gpu"]) if lim.get("gpu") else "N/A",
                f"{lim['mem_mb'] / 1024 / 1024:.0f} TiB",
            ]
        )
    # The reservation qos is not a partition: it is reached by adding --reservation=<name> to a job
    # that still names one of the four partitions above.
    rows.append(
        [
            "reservation<br>(needs `--reservation`)",
            f"`{reservation_qos['qos']}`",
            "min(partition limit, reservation end)",
            "the reserved nodes",
            "16384",
            "1024",
            "1024 TiB",
        ]
    )
    # The total adds the four partitions only. The reservation row is left out of it: its caps are
    # set high enough to never bind, so adding them in would report a limit nobody actually has.
    totals = [sum(int(r[i]) for r in rows[:-1] if r[i] != "N/A") for i in (3, 4, 5)]
    mem_tib = sum(m["per_user_limits_live"]["mem_mb"] for m in partitions) / 1024 / 1024
    rows.append(
        [
            "**total<br>(the four partitions)**",
            "N/A",
            "N/A",
            f"**{totals[0]}**",
            f"**{totals[1]}**",
            f"**{totals[2]}**",
            f"**{mem_tib:.0f} TiB**",
        ]
    )
    return render_table(headers, rows)


def architecture_rollup(nodes):
    """Group GPU nodes by compute capability, returning rows of (cc, family, partitions, counts)."""
    # before: 50 node dicts, each with gpu_compute_capability / gpu_architecture / gpu_model
    # after:  one row per distinct compute capability, e.g.
    #         (6.1, "Pascal", "gpu, gnolim", 14 nodes, 52 gpus, "GTX 1080, GTX 1080 Ti, Titan Xp")
    buckets = {}
    for n in nodes:
        if not n.get("gpu_count"):
            continue
        cc = n["gpu_compute_capability"]
        b = buckets.setdefault(cc, {"fam": n["gpu_architecture"], "parts": set(), "nodes": 0, "gpus": 0, "models": set()})
        b["parts"].update(n["partitions"])
        b["nodes"] += 1
        b["gpus"] += n["gpu_count"]
        b["models"].add(n["gpu_model"])
    rows = []
    for cc in sorted(buckets):
        b = buckets[cc]
        rows.append(
            [
                f"{cc:.1f}",
                b["fam"],
                ", ".join(sorted(b["parts"])),
                str(b["nodes"]),
                str(b["gpus"]),
                ", ".join(sorted(b["models"])),
            ]
        )
    # Each GPU node has one compute capability, so the node and gpu columns add up to the cluster's
    # GPU nodes and GPU cards. The models cell counts distinct models rather than listing them again.
    models = {n["gpu_model"] for n in nodes if n.get("gpu_count")}
    rows.append(
        [
            "**total**",
            "N/A",
            "N/A",
            f"**{sum(b['nodes'] for b in buckets.values())}**",
            f"**{sum(b['gpus'] for b in buckets.values())}**",
            f"**{len(models)} distinct models**",
        ]
    )
    return rows


def main():
    """Read the node inventory and write the compute-resource markdown document."""
    here = os.path.dirname(os.path.abspath(__file__))
    data = json.load(open(os.path.join(here, "..", "data", "cluster_nodes.json")))
    nodes, partitions = data["nodes"], {p["partition"]: p for p in data["partitions"]}
    corrections = apply_measured_overrides(nodes)

    # Header: say where the numbers came from and that the file is generated, matching the
    # convention of the previous resource document.
    lines = [
        "<!-- Generated by code/generate_school_compute_resource.py from data/cluster_nodes.json.",
        "     Do not edit by hand; rerun the generator. -->",
        "",
        "# School compute resources, partition by partition",
        "",
        f"Node inventory read from live Slurm on {data['generated_at']}, cross-checked against the GPU",
        "catalog of the shared `submit-gpu-sweep` skill. The previous version of this document, and how",
        "this one differs from it, are recorded in `../preivous_resource_loation.md`.",
        "",
        (
            f"The cluster has {data['totals']['node_count']} nodes across the four partitions, "
            f"{data['totals']['cpus_total_slurm']} cpus as Slurm counts them, "
            f"{data['totals']['gpu_count']} GPUs on {data['totals']['gpu_node_count']} nodes, and "
            f"{data['totals']['real_memory_mb'] / 1024 / 1024:.1f} TiB of system memory."
        ),
        "",
        "Three things about the numbers before the tables:",
        "",
        "1. **`cpus counted by slurm` counts hardware threads, not cores.** Every node here sets",
        "   `ThreadsPerCore=2`, so a 32-cpu node has 16 physical cores. A job that asks for one cpu per",
        "   task is charged a whole core unless it passes `--ntasks-per-core=2`.",
        "2. **A `*` on a GPU memory value means the catalog's figure is approximate** and should be",
        "   confirmed with `nvidia-smi --query-gpu=memory.total` on the node before it is relied on.",
        "3. **Nodes with the same hardware share one row.** The `node` column then holds the name range",
        "   the row covers, written in Slurm's own `--nodelist` syntax so it can be pasted into an sbatch",
        "   line (`affogato[06-10]`, or `cortado[01-05,07-10]` when the range has a gap). The next column",
        "   says how many nodes that is, and the `state` column counts the states of those nodes.",
        "   Every hardware figure stays per node, so a folded row reads exactly like a single-node row;",
        "   only the closing **total** row of each table adds the nodes up. A node that is not simply",
        "   running work — down, drained, or reserved — is kept on a row of its own so its name and its",
        "   state stay visible.",
        "",
    ]

    # One section per partition, in the order sinfo lists them.
    for idx, (name, blurb) in enumerate(PARTITIONS, start=1):
        meta = partitions[name]
        part_nodes = sorted([n for n in nodes if name in n["partitions"]], key=lambda n: n["node"])
        rows = [node_row(g) for g in fold_nodes(part_nodes)]
        rows.append(node_total_row(part_nodes))
        lines += [
            f"## Table {idx} — partition `{name}`",
            "",
            f"{blurb.capitalize()}. {partition_summary(meta, part_nodes)}",
            "",
            render_table(NODE_COLUMNS, rows),
            "",
        ]

    # The four partition totals side by side, so the whole-cluster figures sit in one row.
    lines += [
        "## Table 5 — the four partitions added up",
        "",
        "The total row of each table above, gathered here. Every node belongs to exactly one partition,",
        "so the last row is the whole cluster.",
        "",
        render_partition_totals_table(data["partitions"], nodes, data["totals"]),
        "",
    ]

    # The per-user limits table: what one user may hold at once in each partition.
    lines += [
        "## Table 6 — per-user limits, per partition",
        "",
        "What one user may hold at once. A reservation job is admitted above the partition caps, but its",
        "usage still counts into them, so open-partition jobs are submitted first and reservation jobs last.",
        "The total row adds the four partitions only: the caps on the reservation row are set high enough",
        "never to bind, so counting them in would report a limit nobody actually has.",
        "",
        render_limits_table(data["partitions"], data["reservation_qos"]),
        "",
        "## Table 7 — GPU generations present, and which PyTorch build reaches them",
        "",
        "A PyTorch wheel carries compiled code for a fixed list of GPU architectures. Code compiled for",
        "one architecture runs on any card with the same major compute-capability number, so an `sm_86`",
        "build also covers compute capability 8.9, but nothing covers a major number that is absent.",
        "",
        render_table(
            ["compute capability", "architecture", "partitions", "nodes", "gpus", "models"],
            architecture_rollup(nodes),
        ),
        "",
        "Measured on 2026-08-05 with `torch._C._cuda_getArchFlags()`:",
        "",
        "| build | compiled architectures | major numbers covered | GPUs on this cluster it can run |",
        "|---|---|---|---|",
        "| `torch 2.5.1+cu124` (the `exploration` env) | `sm_50 sm_60 sm_70 sm_75 sm_80 sm_86 sm_90` | 5, 6, 7, 8, 9 | every GPU except the 4 RTX 5080 cards |",
        "| `torch 2.10.0+cu128` (in `/u/sl5nw/.local`) | `sm_70 sm_75 sm_80 sm_86 sm_90 sm_100 sm_120` | 7, 8, 9, 10, 12 | no Pascal and no Maxwell, so none of `gnolim` and none of lynx01-07 or affogato13-15 |",
        "",
        "The second row is a live hazard, not a hypothetical: `/u/sl5nw/.local/lib/python3.11/site-packages`",
        "sits ahead of a shared env's own `site-packages` on `sys.path`, so a job run as `sl5nw` imports the",
        "2.10 build unless it exports `PYTHONNOUSERSITE=1`. Every script in this folder sets it.",
        "",
    ]

    # Anywhere a card contradicted the shared catalog, say so with both values.
    if corrections:
        lines += [
            "## Where a card contradicted the shared catalog",
            "",
            "These rows were read off the hardware and replace the catalog's values above. The catalog",
            "(`submit-gpu-sweep/server_introduction/`) feeds the group's GPU packing tools, so a wrong",
            "entry there decides which nodes a sweep may use — which is not hypothetical: the wrong",
            "`titanx03` entry excluded that node from a sweep whose floor it actually clears.",
            "",
            render_table(["node", "field", "catalog says", "the card says"],
                         [[f"`{n}`", f, str(o), f"**{v}**"] for n, f, o, v in corrections]),
            "",
        ]
        for node, spec in MEASURED_OVERRIDES.items():
            lines += [f"Source for `{node}`: {spec['source']}", ""]

    out_path = os.path.join(here, "..", "2026-08-05_school_compute_resource.md")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"wrote {os.path.abspath(out_path)} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
