"""Write one Slurm submission script per probe and per measurement job.

A measurement job = one node class at one processor count; it measures every copy count inside
its own wall-clock deadline. A probe job = one node class, one short trainer iteration, to find
out whether the installed JAX reaches that card's generation at all.

Jobs are pinned to a node of their class with `--nodelist`, because the whole point is to
measure a named piece of hardware; without the pin Slurm would place the job wherever it liked
and the measurement would not know what it measured.
"""
import argparse
import stat
import subprocess
from pathlib import Path

from node_classes import cpu_counts_for, load_classes

RUN = Path(__file__).resolve().parent.parent
ENV = RUN / "code" / "worker_env.sh"

PROBE = """#!/bin/bash
#SBATCH --job-name=gpuprobe-{name}
#SBATCH --partition={partition}
#SBATCH --nodelist={node}
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00{reservation}
#SBATCH --output={run}/slurm/logs/probe__{name}.out
#SBATCH --error={run}/slurm/logs/probe__{name}.err
set -euo pipefail
source {env}
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv
"$SURVEY_PYTHON" {run}/code/probe_gpu.py \\
    --node-class {name} --out {run}/data/probe/{name}.json
"""

MEASURE = """#!/bin/bash
#SBATCH --job-name=gputhr-{name}-c{cpus}
#SBATCH --partition={partition}
#SBATCH --nodelist={node}
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem_gb}G
#SBATCH --time=00:45:00{reservation}
#SBATCH --output={run}/slurm/logs/{name}__cpus-{cpus}.out
#SBATCH --error={run}/slurm/logs/{name}__cpus-{cpus}.err
set -euo pipefail
source {env}
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv
"$SURVEY_PYTHON" {run}/code/run_job.py \\
    --node-class {name} --cpus {cpus} --style {style} --deadline-minutes 27 \\
    --out-dir {run}/data/throughput --log-dir {run}/slurm/logs \\
    --python "$SURVEY_PYTHON"
"""


def reserved_nodes():
    """The nodes covered by this user's own Slurm reservation, and the reservation's name.

    A reservation carries IGNORE_JOBS and SPEC_NODES, so a job WITHOUT `--reservation` cannot run
    on a reserved node at all — it pends forever with "May be reserved for other job". The name
    changes over time and is therefore read live, never hardcoded.

    before: `scontrol show reservation -o` lists ReservationName=sl5nw_156 ... Nodes=jaguar03,puma01
    after:  ({"jaguar03", "puma01"}, "sl5nw_156")
    """
    out = subprocess.run(["scontrol", "show", "reservation", "-o"],
                         capture_output=True, text=True).stdout
    for line in out.splitlines():
        fields = dict(f.split("=", 1) for f in line.split() if "=" in f)
        if fields.get("Users", "") and "sl5nw" in fields["Users"]:
            nodes = subprocess.run(["scontrol", "show", "hostnames", fields["Nodes"]],
                                   capture_output=True, text=True).stdout.split()
            return set(nodes), fields["ReservationName"]
    return set(), None


# the largest host memory any finished job of this survey actually used is 7.7 GiB (sacct MaxRSS
# over the first 51 jobs, at 4,096 copies); 16 GB is that with room to spare. Asking for a
# quarter of the node instead — 32 GB on a 125 GiB node, 64 on a serval — was what kept jobs
# queued on nodes that had a free card but only 29 GiB of memory left, so the ask is a measured
# constant now rather than a fraction of hardware the job never touches.
HOST_MEMORY_GB = 16


def free_capacity(node):
    """Processors, cards and memory a node has left right now, read live from Slurm.

    before: `scontrol show node cheetah09` reports CfgTRES cpu=38,mem=500G,gres/gpu=4 and
            AllocTRES cpu=16,mem=128G,gres/gpu=2
    after:  {"cpus": 22, "gpus": 2, "mem_gb": 372} — enough for one 8-processor job, so this
            node is chosen over cheetah08, which has all four of its cards taken
    """
    text = subprocess.run(["scontrol", "show", "node", node],
                          capture_output=True, text=True).stdout
    def tres(field):
        block = text.split(f"{field}=", 1)[1].split()[0] if f"{field}=" in text else ""
        parts = dict(p.split("=") for p in block.split(",") if "=" in p)
        return (int(parts.get("cpu", 0)),
                int(parts.get("gres/gpu", 0)),
                int(parts.get("mem", "0G").rstrip("GM").split(".")[0] or 0))
    cfg_cpu, cfg_gpu, cfg_mem = tres("CfgTRES")
    alloc_cpu, alloc_gpu, alloc_mem = tres("AllocTRES")
    state = text.split("State=", 1)[1].split()[0] if "State=" in text else ""
    return {"cpus": cfg_cpu - alloc_cpu, "gpus": cfg_gpu - alloc_gpu,
            "mem_gb": cfg_mem - alloc_mem, "state": state, "node": node}


def pick_node(node_class, cpus):
    """The node of this class that can start the job soonest — the freest one that still fits.

    Nodes of one class are identical hardware, so which one runs the measurement does not change
    the number; it only decides whether the job waits. A node in maintenance is never chosen.
    """
    options = [free_capacity(n) for n in node_class["nodes"]]
    runnable = [o for o in options
                if "MAINTENANCE" not in o["state"] and o["gpus"] >= 1
                and o["cpus"] >= cpus and o["mem_gb"] >= HOST_MEMORY_GB]
    best = max(runnable or options, key=lambda o: (o["gpus"], o["cpus"], o["mem_gb"]))
    return best["node"]


def write(path, text):
    """Write an executable submission script."""
    path.write_text(text)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


def main():
    """Generate the probe and measurement scripts for every node class."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--style", default="full_batch")
    args = ap.parse_args()
    out = RUN / "slurm" / "scripts"
    out.mkdir(parents=True, exist_ok=True)

    reserved, reservation_name = reserved_nodes()
    n_probe = n_measure = 0
    for cls in load_classes():
        for cpus in cpu_counts_for(cls):
            # any node of the class measures the same hardware, so pick whichever can start
            # soonest; the choice is remade every time the scripts are regenerated
            node = pick_node(cls, cpus)
            # a reserved node needs the reservation named, and the reservation's own qos, or the
            # job is charged to the open-partition caps and pends on them even though the
            # reservation would admit it
            booking = (f"\n#SBATCH --reservation={reservation_name}\n#SBATCH --qos=csresnolim"
                       if node in reserved else "")
            common = dict(name=cls["name"], partition=cls["partition"], node=node,
                          run=RUN, env=ENV, style=args.style, reservation=booking)
            write(out / f"{cls['name']}__cpus-{cpus}.slurm",
                  MEASURE.format(cpus=cpus, mem_gb=HOST_MEMORY_GB, **common))
            n_measure += 1
        write(out / f"probe__{cls['name']}.slurm", PROBE.format(**common))
        n_probe += 1
    print(f"wrote {n_probe} probe scripts and {n_measure} measurement scripts to {out}")


if __name__ == "__main__":
    main()
