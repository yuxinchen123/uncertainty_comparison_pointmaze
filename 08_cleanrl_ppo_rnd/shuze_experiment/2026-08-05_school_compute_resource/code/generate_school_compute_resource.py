"""Render the school compute-resource document from the node inventory JSON.

Reads ../data/cluster_nodes.json (written by build_cluster_nodes_json.py from live Slurm plus the
shared GPU catalog) and writes ../2026-08-05_school_compute_resource.md: one table per partition
(cpu, gpu, nolim, gnolim), one row per node, plus the per-user limit table and the GPU-generation
compatibility section.

Run:
    PYTHONNOUSERSITE=1 /p/rlprojects/RND/.venvs/exploration/bin/python \
        generate_school_compute_resource.py
"""

import json
import os
from datetime import datetime

# The four partitions, in the order sinfo lists them. Each entry is (partition name, one-line
# description used as the table caption).
PARTITIONS = [
    ("cpu", "no GPUs; the largest core pool on the cluster"),
    ("gpu", "every GPU node that is not in gnolim; a 4-day time limit"),
    ("nolim", "no GPUs, a 20-day time limit, small core pool"),
    ("gnolim", "older GPU nodes with a 20-day time limit"),
]

# Column headers of the per-partition node tables. Kept literal: each one names exactly what the
# cell holds. "N/A" fills the GPU columns of a node that has no GPU.
NODE_COLUMNS = [
    "node",
    "state",
    "cpu model",
    "sockets x cores x threads",
    "cpus counted by slurm",
    "system memory",
    "gpus per node",
    "gpu model",
    "gpu memory per card",
    "gpu architecture (year)",
    "gpu compute capability",
]


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


def node_row(node):
    """Build one markdown table row for a node, as a list of cell strings in NODE_COLUMNS order."""
    # Sockets/cores/threads come straight from scontrol; they multiply out to "cpus counted by
    # slurm", which counts hardware threads, not cores.
    # before: sockets=2, cores_per_socket=8, threads_per_core=2
    # after:  "2 x 8 x 2"  and  cpus counted by slurm = 32
    geometry = "{} x {} x {}".format(
        node.get("sockets"), node.get("cores_per_socket"), node.get("threads_per_core")
    )
    has_gpu = node.get("gpu_count", 0) > 0
    return [
        f"`{node['node']}`",
        node.get("state", "").lower(),
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


def render_table(headers, rows):
    """Render a markdown table from a header list and a list of row-cell lists."""
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


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
            f"{data['totals']['real_memory_mb'] / 1024 / 1024:.0f} TiB of system memory."
        ),
        "",
        "Two things about the numbers before the tables:",
        "",
        "1. **`cpus counted by slurm` counts hardware threads, not cores.** Every node here sets",
        "   `ThreadsPerCore=2`, so a 32-cpu node has 16 physical cores. A job that asks for one cpu per",
        "   task is charged a whole core unless it passes `--ntasks-per-core=2`.",
        "2. **A `*` on a GPU memory value means the catalog's figure is approximate** and should be",
        "   confirmed with `nvidia-smi --query-gpu=memory.total` on the node before it is relied on.",
        "",
    ]

    # One section per partition, in the order sinfo lists them.
    for idx, (name, blurb) in enumerate(PARTITIONS, start=1):
        meta = partitions[name]
        part_nodes = sorted([n for n in nodes if name in n["partitions"]], key=lambda n: n["node"])
        lines += [
            f"## Table {idx} — partition `{name}`",
            "",
            f"{blurb.capitalize()}. {partition_summary(meta, part_nodes)}",
            "",
            render_table(NODE_COLUMNS, [node_row(n) for n in part_nodes]),
            "",
        ]

    # The per-user limits table: what one user may hold at once in each partition.
    lines += [
        "## Table 5 — per-user limits, per partition",
        "",
        "What one user may hold at once. A reservation job is admitted above the partition caps, but its",
        "usage still counts into them, so open-partition jobs are submitted first and reservation jobs last.",
        "",
        render_limits_table(data["partitions"], data["reservation_qos"]),
        "",
        "## Table 6 — GPU generations present, and which PyTorch build reaches them",
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
