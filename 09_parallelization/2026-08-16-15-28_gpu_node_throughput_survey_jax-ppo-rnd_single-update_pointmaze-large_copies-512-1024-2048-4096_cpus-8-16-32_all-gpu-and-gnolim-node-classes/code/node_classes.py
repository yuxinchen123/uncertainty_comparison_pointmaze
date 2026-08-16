"""The node classes this survey covers, and the processor counts each one can actually give.

One node class = one row of the cluster's machine-readable node-class file
(`school_compute_resource/server_introduction.json` in the shared uva-submit-gpu-sweep skill):
nodes identical in graphics-card type, cards per node, card memory, processor type, processor
threads and partition. Testing one node per class is what "test each configuration once" means
here — two nodes in the same class are the same machine twice.
"""
import json
from pathlib import Path

SERVER_INTRODUCTION = Path(
    "/p/rlprojects/.claude/skills/uva-submit-gpu-sweep/school_compute_resource/"
    "server_introduction.json")

# the processor counts the user asked for, in the order they are measured
REQUESTED_CPUS = (8, 16, 32)

# the copy counts the user asked for, measured smallest first so a card that runs out of memory
# at one count has already produced every smaller one
COPY_COUNTS = (512, 1024, 2048, 4096)


def load_classes(partitions=("gpu", "gnolim")):
    """Every node class of the given partitions, highest card capability first."""
    data = json.loads(SERVER_INTRODUCTION.read_text())
    classes = [c for c in data["classes"] if c["partition"] in partitions]
    return sorted(classes, key=lambda c: (c.get("capability_rank", 99), c["name"]))


def cpu_counts_for(node_class):
    """The processor counts to measure on one class: the requested ones it can allocate, and its
    own maximum when 32 is out of reach — so no class is left with fewer than three points
    unless its hardware genuinely has fewer.

    A node reserves one core (two hardware threads) for the system, so `cpu_alloc_threads` is
    below the thread count in the hardware table and a request above it never starts.

    before: cpu_alloc_threads = 30 (a 32-thread node), requested (8, 16, 32)
    after:  (8, 16, 30) — 32 is unallocatable, so the class's own maximum stands in for it
    before: cpu_alloc_threads = 62, requested (8, 16, 32)
    after:  (8, 16, 32) — every requested count fits
    before: cpu_alloc_threads = 14 (jaguar05), requested (8, 16, 32)
    after:  (8, 14) — only one count above 8 exists on this hardware
    """
    cap = node_class["cpu_alloc_threads"]
    counts = [c for c in REQUESTED_CPUS if c <= cap]
    if len(counts) < len(REQUESTED_CPUS) and cap > (counts[-1] if counts else 0):
        counts.append(cap)   # the class cannot reach 32; its own maximum stands in for it
    return tuple(counts)


def survey_cells():
    """Every (node class, processor count) pair this survey submits as one Slurm job."""
    return [(c, n) for c in load_classes() for n in cpu_counts_for(c)]


if __name__ == "__main__":
    total_jobs = 0
    for cls in load_classes():
        counts = cpu_counts_for(cls)
        total_jobs += len(counts)
        print(f"{cls['name']:16} {cls['partition']:7} {cls['display_name']:16} "
              f"cc {cls['compute_capability']:<5} {cls['gpu_mem_mb']:>6} MB  "
              f"{cls['cpu_type']:18} alloc {cls['cpu_alloc_threads']:>3}  cpus {counts}")
    print(f"\n{len(load_classes())} node classes, {total_jobs} jobs, "
          f"{total_jobs * len(COPY_COUNTS)} measurement cells")
