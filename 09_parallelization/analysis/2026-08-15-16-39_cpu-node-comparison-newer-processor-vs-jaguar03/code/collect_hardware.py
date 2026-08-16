"""Turn the per-node hardware logs into one JSON the report module can read.

The two jobs each wrote two text files: what the operating system says about the processor
(`hardware_<node>.txt`, from lscpu and friends) and what the probe measured (`probe_<node>.txt`,
from cpu_probe.c). This script pulls the numbers out of both into `data/node_hardware.json`, so
the report never has to parse text and no hardware figure is ever typed in by hand.

Every field records where it came from, because the question the report answers is partly "how do
you know": `source` is either the command that printed it or the probe that measured it.

Run: python collect_hardware.py
"""
import json
import re
from pathlib import Path

RUN = Path(__file__).resolve().parent.parent
LOGS = RUN / "logs"
DATA = RUN / "data"


def first_json_object(text, key):
    """The first {...} block in `text` that contains `key`, parsed."""
    # the probe prints one JSON object per invocation; the full object is the one carrying the
    # latency measurements, while the short --clock-only ones carry only the clock
    # before: '...idle =====\n{\n "clock_ghz...": 3.59,\n "latency_ns...": {...}\n}\n=====...'
    # after:  {'clock_ghz_dependent_add_chain': 3.59, 'latency_ns_per_dependent_access': {...}}
    for match in re.finditer(r"\{(?:[^{}]|\{[^{}]*\})*\}", text, re.S):
        blob = match.group(0)
        if key in blob:
            return json.loads(blob)
    raise ValueError(f"no JSON object containing {key!r} was found")


def lscpu_value(text, field):
    """The value of one `lscpu` field, e.g. 'Model name' -> 'Intel(R) Xeon(R) Gold 6334 ...'."""
    # before: 'Model name:                              Intel(R) Xeon(R) Gold 6334 CPU @ 3.60GHz'
    # after:  'Intel(R) Xeon(R) Gold 6334 CPU @ 3.60GHz'
    match = re.search(rf"^{re.escape(field)}:\s+(.+?)\s*$", text, re.M)
    if match is None:
        raise ValueError(f"lscpu field {field!r} was not in the log")
    return match.group(1)


def cache_row(text, level):
    """One row of `lscpu -C` as (per-instance size, total size, number of instances)."""
    # before: 'L2       1.3M      20M   20 Unified         2  1024        1             64'
    # after:  ('1.3M', '20M')  -- instances come from the lscpu summary line, not this row
    # the two size fields must look like sizes, so that the plain lscpu summary line a few lines
    # earlier ('L2 cache:   20 MiB (16 instances)') cannot be mistaken for the table row
    size = r"\d+(?:\.\d+)?[KMG]"
    match = re.search(rf"^{level}\s+({size})\s+({size})\s", text, re.M)
    if match is None:
        raise ValueError(f"cache level {level!r} was not in the log")
    return match.group(1), match.group(2)


def cache_instances(text, level):
    """How many separate caches of one level the processor has, from the lscpu summary line."""
    # before: 'L3 cache:                                36 MiB (2 instances)'
    # after:  2
    match = re.search(rf"^{level} cache:\s+\S+\s+\S+\s+\((\d+) instances?\)", text, re.M)
    if match is None:
        raise ValueError(f"instance count for {level!r} was not in the log")
    return int(match.group(1))


def loaded_clock_ghz(probe_text):
    """Mean of the clock samples taken while the training workload occupied every other core."""
    # before: 'sample 1 at ...: {"clock_ghz_dependent_add_chain": 3.563}' x 5
    # after:  3.5624
    # the timestamp between "at" and the JSON contains colons of its own, so the pattern skips to
    # the end of the line rather than to the first colon
    values = [float(v) for v in
              re.findall(r'sample \d+ at [^\n]*?\{"clock_ghz_dependent_add_chain": ([\d.]+)\}',
                         probe_text)]
    if not values:
        raise ValueError("no loaded-clock samples were in the probe log")
    return sum(values) / len(values)


def memory_channels(mem_text):
    """Number of memory channels per socket and the memory type, from the EDAC labels."""
    # before: 'dimm0 label=CPU_SrcID#0_MC#0_Chan#0_DIMM#0 size=65536 type=Unbuffered-DDR4'
    # after:  (8, 'DDR4', 16, 65536)  -- 8 channels on socket 0, DDR4, 16 DIMMs total, 64 GiB each
    labels = set(re.findall(r"label=(CPU_SrcID#\d+_MC#\d+_Chan#\d+_DIMM#\d+)", mem_text))
    if not labels:
        return None
    per_socket = {}
    for label in labels:
        socket = re.search(r"SrcID#(\d+)", label).group(1)
        channel = re.search(r"MC#(\d+)_Chan#(\d+)", label).groups()
        per_socket.setdefault(socket, set()).add(channel)
    types = set(re.findall(r"type=(\S+)", mem_text))
    sizes = {int(s) for s in re.findall(r"size=(\d+)", mem_text)}
    return {"channels_per_socket": max(len(v) for v in per_socket.values()),
            "sockets_populated": len(per_socket),
            "memory_type": sorted(types)[0] if types else None,
            "dimms_total": len(labels),
            "dimm_size_mib": max(sizes) if sizes else None}


def collect(node):
    """Every hardware figure for one node, each tagged with the command that produced it."""
    hw = (LOGS / f"hardware_{node}.txt").read_text()
    probe = (LOGS / f"probe_{node}.txt").read_text()
    mem_path = LOGS / f"memory_{node}.txt"
    idle = first_json_object(probe, "latency_ns_per_dependent_access")

    # the caches: lscpu -C gives the size of one cache, the summary line gives how many there are,
    # and cores-per-cache follows from the core count divided by the number of caches
    cores = int(lscpu_value(hw, "Core(s) per socket")) * int(lscpu_value(hw, "Socket(s)"))
    l3_instances = cache_instances(hw, "L3")
    out = {
        "model_name": lscpu_value(hw, "Model name"),
        "sockets": int(lscpu_value(hw, "Socket(s)")),
        "cores_per_socket": int(lscpu_value(hw, "Core(s) per socket")),
        "threads_per_core": int(lscpu_value(hw, "Thread(s) per core")),
        "physical_cores": cores,
        "hardware_threads": int(lscpu_value(hw, "CPU(s)")),
        "numa_nodes": int(lscpu_value(hw, "NUMA node(s)")),
        "cpu_family_model_stepping": (f"{lscpu_value(hw, 'CPU family')}/"
                                      f"{lscpu_value(hw, 'Model')}/"
                                      f"{lscpu_value(hw, 'Stepping')}"),
        "l1d_per_core": cache_row(hw, "L1d")[0],
        "l1i_per_core": cache_row(hw, "L1i")[0],
        "l2_per_core": cache_row(hw, "L2")[0],
        "l3_per_instance": cache_row(hw, "L3")[0],
        "l3_total": cache_row(hw, "L3")[1],
        "l3_instances": l3_instances,
        "cores_sharing_one_l3": cores / l3_instances,
        "clock_ghz_one_core_busy": idle["clock_ghz_dependent_add_chain"],
        "clock_ghz_every_core_busy": loaded_clock_ghz(probe),
        "latency_ns": idle["latency_ns_per_dependent_access"],
        "read_bandwidth_gb_per_s_one_core": idle["read_bandwidth_gb_per_s_one_core"],
        "vector_instruction_set_used_by_torch":
            re.search(r"cpu capability (\S+)", probe).group(1),
        "sources": {
            "model_name, core and thread counts, cache sizes": "lscpu on the node",
            "clock, memory access time, memory read rate": "cpu_probe.c run on the node",
            "vector instruction set": "torch.backends.cpu.get_cpu_capability() on the node",
            "memory channels and type": "EDAC labels under /sys/devices/system/edac on the node",
        },
    }
    # the nominal clock printed inside the model name, where the manufacturer put one
    nominal = re.search(r"@ ([\d.]+)GHz", out["model_name"])
    out["nominal_clock_ghz_in_model_name"] = float(nominal.group(1)) if nominal else None
    if mem_path.exists():
        out["memory"] = memory_channels(mem_path.read_text())
    return out


def main():
    DATA.mkdir(exist_ok=True)
    nodes = sorted(p.name[len("probe_"):-len(".txt")] for p in LOGS.glob("probe_*.txt"))
    result = {node: collect(node) for node in nodes}
    (DATA / "node_hardware.json").write_text(json.dumps(result, indent=1))
    for node, facts in result.items():
        print(f"{node}: {facts['model_name']}, {facts['physical_cores']} cores, "
              f"{facts['hardware_threads']} threads, "
              f"clock {facts['clock_ghz_one_core_busy']:.2f} GHz idle / "
              f"{facts['clock_ghz_every_core_busy']:.2f} GHz loaded, "
              f"{facts['vector_instruction_set_used_by_torch']}")
    print(f"wrote {DATA / 'node_hardware.json'}")


if __name__ == "__main__":
    main()
