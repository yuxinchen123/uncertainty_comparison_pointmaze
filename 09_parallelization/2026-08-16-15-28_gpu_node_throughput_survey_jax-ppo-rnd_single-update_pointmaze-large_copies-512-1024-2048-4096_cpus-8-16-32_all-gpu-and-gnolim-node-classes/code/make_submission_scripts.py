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


def host_memory_gb(node_class):
    """How much system memory one job asks for: enough for the largest run, never the whole node.

    The trainer holds its arrays on the card; the host side carries only the program, the
    parameters at build time and the result records. 64 GB is generous for that and leaves the
    node's other cards usable by other jobs — except where the node has less than that in total,
    where the job takes a quarter of what the node has.

    before: sys_mem_mb = 64000 (an ai01-04 node) -> after: 16 GB asked
    before: sys_mem_mb = 1500000 (a serval node) -> after: 64 GB asked
    """
    return min(64, max(8, node_class["sys_mem_mb"] // 1000 // 4))


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
        # the first node of the class stands for the class; the submitter may swap in another
        # node of the same class if this one is busy, since they are identical by construction
        node = cls["nodes"][0]
        # a reserved node needs the reservation named, and the reservation's own qos, or the job
        # is charged to the open-partition caps and pends on them even though it would be admitted
        booking = (f"\n#SBATCH --reservation={reservation_name}\n#SBATCH --qos=csresnolim"
                   if node in reserved else "")
        common = dict(name=cls["name"], partition=cls["partition"], node=node,
                      run=RUN, env=ENV, style=args.style, reservation=booking)
        write(out / f"probe__{cls['name']}.slurm", PROBE.format(**common))
        n_probe += 1
        for cpus in cpu_counts_for(cls):
            write(out / f"{cls['name']}__cpus-{cpus}.slurm",
                  MEASURE.format(cpus=cpus, mem_gb=host_memory_gb(cls), **common))
            n_measure += 1
    print(f"wrote {n_probe} probe scripts and {n_measure} measurement scripts to {out}")


if __name__ == "__main__":
    main()
