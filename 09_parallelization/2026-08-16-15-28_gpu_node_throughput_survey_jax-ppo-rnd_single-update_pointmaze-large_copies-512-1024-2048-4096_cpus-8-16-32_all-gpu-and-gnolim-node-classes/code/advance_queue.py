"""Submit the measurement jobs that are due, and report what the survey is still missing.

Two rules decide what may be submitted:

1. **One live job per node class.** Two of this survey's own jobs on the same machine would sit
   on different cards but share its memory bandwidth and its processors, so each would measure
   the other's interference as if it were the hardware. A class therefore carries at most one
   job that is pending or running, and its next processor count is submitted only once the
   previous one has ended.
2. **The per-user caps of each partition.** Running jobs count against them
   (gpu: 40 cards / 400 processors, gnolim: 20 cards / 80 processors); pending jobs do not.
   The owner of a sweep leaves about 16 processors of headroom under each cap for work submitted
   by hand.

Job identity is read only from this survey's own id file — never by enumerating the user's jobs,
which are shared with other sessions.
"""
import argparse
import json
import subprocess
from pathlib import Path

from node_classes import cpu_counts_for, load_classes

RUN = Path(__file__).resolve().parent.parent
ID_FILE = RUN / "slurm" / "submitted_jobids.txt"
SCRIPTS = RUN / "slurm" / "scripts"
RESULTS = RUN / "data" / "throughput"
OWNER_HEADROOM_CPUS = 16
CAPS = {"gpu": {"gpus": 40, "cpus": 400}, "gnolim": {"gpus": 20, "cpus": 80}}


def own_job_states():
    """State and name of every job this survey has ever submitted, from its own id file only."""
    ids = [line.split()[0] for line in ID_FILE.read_text().splitlines() if line.strip()]
    if not ids:
        return {}
    out = subprocess.run(
        ["sacct", "-j", ",".join(ids), "-X", "--noheader", "--parsable2",
         "--format=JobID,JobName,State"], capture_output=True, text=True).stdout
    states = {}
    for line in out.splitlines():
        job_id, name, state = line.split("|")[:3]
        states[job_id] = {"name": name, "state": state.split()[0]}
    return states


def live_cells(states):
    """The (node class, processor count) cells that currently hold a pending or running job.

    before: a job named `gputhr-lynx02-04-c16` in state RUNNING
    after:  ("lynx02-04", 16) is live, so this class submits nothing further this tick
    """
    live = set()
    for job in states.values():
        if job["state"] in ("PENDING", "RUNNING", "REQUEUED", "SUSPENDED", "CONFIGURING"):
            if job["name"].startswith("gputhr-"):
                class_name, cpus = job["name"][len("gputhr-"):].rsplit("-c", 1)
                live.add((class_name, int(cpus)))
    return live


def finished_cells():
    """The cells that already produced a result file — never measured a second time."""
    return {(f.name.split("__")[0], int(f.name.split("__cpus-")[1].split("__")[0]))
            for f in RESULTS.glob("*__cpus-*__*.json") if ".cell." not in f.name}


def running_usage(states, node_class_of):
    """Cards and processors this survey currently has RUNNING, per partition."""
    usage = {p: {"gpus": 0, "cpus": 0} for p in CAPS}
    for job in states.values():
        if job["state"] != "RUNNING" or not job["name"].startswith("gputhr-"):
            continue
        class_name, cpus = job["name"][len("gputhr-"):].rsplit("-c", 1)
        partition = node_class_of[class_name]["partition"]
        usage[partition]["gpus"] += 1
        usage[partition]["cpus"] += int(cpus)
    return usage


def main():
    """Submit each class's next processor count, within the caps."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--style", default="full_batch")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    classes = load_classes()
    node_class_of = {c["name"]: c for c in classes}
    states = own_job_states()
    live, done = live_cells(states), finished_cells()
    usage = running_usage(states, node_class_of)

    submitted, waiting, complete = [], [], []
    for cls in classes:
        counts = cpu_counts_for(cls)
        # the class's next unmeasured processor count, smallest first
        todo = [n for n in counts if (cls["name"], n) not in done]
        if not todo:
            complete.append(cls["name"])
            continue
        if any(cell[0] == cls["name"] for cell in live):
            waiting.append(f"{cls['name']} (a job is already live)")
            continue
        cpus, partition = todo[0], cls["partition"]
        # room under this partition's caps, with the owner's headroom kept free
        room = (usage[partition]["gpus"] + 1 <= CAPS[partition]["gpus"]
                and usage[partition]["cpus"] + cpus
                <= CAPS[partition]["cpus"] - OWNER_HEADROOM_CPUS)
        if not room:
            waiting.append(f"{cls['name']} cpus {cpus} ({partition} cap reached this tick)")
            continue
        script = SCRIPTS / f"{cls['name']}__cpus-{cpus}.slurm"
        if args.dry_run:
            submitted.append(f"[dry run] {script.name}")
            continue
        job_id = subprocess.run(["sbatch", "--parsable", str(script)],
                                capture_output=True, text=True, check=True).stdout.strip()
        with ID_FILE.open("a") as fh:
            fh.write(f"{job_id}  {script.name}\n")
        usage[partition]["gpus"] += 1
        usage[partition]["cpus"] += cpus
        submitted.append(f"{job_id}  {script.name}")

    print(f"submitted {len(submitted)}:")
    for s in submitted:
        print("  ", s)
    print(f"waiting {len(waiting)}:")
    for w in waiting:
        print("  ", w)
    print(f"classes with every processor count measured: {len(complete)}/{len(classes)}")
    print(json.dumps(usage))


if __name__ == "__main__":
    main()
