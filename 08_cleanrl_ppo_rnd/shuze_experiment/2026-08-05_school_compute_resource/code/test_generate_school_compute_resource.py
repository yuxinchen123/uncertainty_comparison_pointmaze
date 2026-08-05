"""Tests for the node folding and total rows of generate_school_compute_resource.py.

Run:
    PYTHONNOUSERSITE=1 /p/rlprojects/RND/.venvs/exploration/bin/python -m pytest \
        test_generate_school_compute_resource.py -q
"""

import pytest

from generate_school_compute_resource import (
    NODE_COLUMNS,
    fold_nodes,
    format_node_range,
    format_states,
    node_row,
    node_total_row,
    split_node_name,
)


def make_node(name, state="IDLE", cpus=32, memory_mb=128000, gpus=0, **extra):
    """Build the subset of an inventory node dict that the folding code reads."""
    node = {
        "node": name,
        "state": state,
        "cpu_model": "Intel Skylake",
        "sockets": 2,
        "cores_per_socket": 8,
        "threads_per_core": 2,
        "cpus_total_slurm": cpus,
        "real_memory_mb": memory_mb,
        "gpu_count": gpus,
        "gpu_model": None,
        "gpu_memory_gib_per_card": None,
        "gpu_memory_label": None,
        "gpu_architecture": None,
        "gpu_architecture_year": None,
        "gpu_compute_capability": None,
    }
    node.update(extra)
    return node


def test_split_node_name_reads_prefix_number_and_padding():
    """A name splits into its letter prefix, its number, and the width the number is written in."""
    assert split_node_name("affogato07") == ("affogato", 7, 2)
    assert split_node_name("slurm1") == ("slurm", 1, 1)
    assert split_node_name("hydro") == ("hydro", None, 0)


def test_format_node_range_covers_single_run_and_gap():
    """One name stays itself; a run becomes a bracket range; a gap splits into two pieces."""
    assert format_node_range(["hydro"]) == "hydro"
    assert format_node_range([f"affogato{i:02d}" for i in range(6, 11)]) == "affogato[06-10]"
    names = [f"cortado{i:02d}" for i in list(range(1, 6)) + list(range(7, 11))]
    assert format_node_range(names) == "cortado[01-05,07-10]"
    assert format_node_range(["ai05", "ai10"]) == "ai[05,10]"


def test_format_node_range_rejects_mixed_prefixes():
    """Folding names from two different machine families is a bug, not something to paper over."""
    with pytest.raises(ValueError):
        format_node_range(["lynx01", "cortado02"])


def test_fold_nodes_groups_identical_hardware_and_keeps_odd_states_apart():
    """Same hardware folds into one row; a node that is down or reserved keeps a row of its own."""
    # before: 4 nodes — three identical and running work, one identical but down
    # after:  two groups — [cortado01, cortado02, cortado04] and [cortado03]
    nodes = [
        make_node("cortado01", state="MIXED"),
        make_node("cortado02", state="IDLE"),
        make_node("cortado03", state="DOWN+NOT_RESPONDING"),
        make_node("cortado04", state="ALLOCATED"),
    ]
    groups = fold_nodes(nodes)
    assert [[n["node"] for n in g] for g in groups] == [
        ["cortado01", "cortado02", "cortado04"],
        ["cortado03"],
    ]


def test_fold_nodes_separates_different_hardware():
    """A different memory size or core count is a different machine, so it does not fold."""
    nodes = [
        make_node("bigcat01", memory_mb=1500000),
        make_node("bigcat02", memory_mb=1500000),
        make_node("bigcat03", memory_mb=1400000),
        make_node("bigcat04", cpus=64, memory_mb=1500000),
    ]
    groups = fold_nodes(nodes)
    assert [[n["node"] for n in g] for g in groups] == [
        ["bigcat01", "bigcat02"],
        ["bigcat03"],
        ["bigcat04"],
    ]


def test_format_states_counts_a_group_and_names_a_single_node():
    """A folded group reports how many nodes are in each state; a single node reports its state."""
    group = [make_node(f"n{i}", state=s) for i, s in enumerate(["MIXED", "IDLE", "MIXED"])]
    assert format_states(group) == "2 mixed, 1 idle"
    assert format_states([make_node("n0", state="ALLOCATED")]) == "allocated"


def test_node_row_reports_per_node_figures_for_a_folded_group():
    """The hardware cells of a folded row stay per node; only the node-count cell counts nodes."""
    group = [make_node(f"struct0{i}", cpus=28, memory_mb=128000) for i in range(1, 4)]
    row = node_row(group)
    assert len(row) == len(NODE_COLUMNS)
    assert row[0] == "`struct[01-03]`"
    assert row[1] == "3"
    assert row[5] == "28"
    assert row[6] == "125 GiB"


def test_node_total_row_adds_nodes_cpus_memory_and_gpus():
    """The closing row sums the node count, the cpus, the memory, and the gpus."""
    nodes = [
        make_node("ai07", cpus=32, memory_mb=128000, gpus=4),
        make_node("ai08", cpus=32, memory_mb=128000, gpus=4),
        make_node("jinx01", cpus=24, memory_mb=220160, gpus=2),
    ]
    row = node_total_row(nodes)
    assert len(row) == len(NODE_COLUMNS)
    assert row[0] == "**total**"
    assert row[1] == "**3**"
    assert row[5].startswith("**88**")
    assert row[6].startswith("**0.5 TiB**")
    assert row[7].startswith("**10**")
