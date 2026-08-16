"""One Slurm job: every copy count on one node class at one processor count.

Each copy count runs as its own process, so a card that runs out of memory at 4,096 copies
still reports 512, 1,024 and 2,048. The whole job is held under a wall-clock deadline (the
user's 30-minute cap) by giving each remaining copy count an equal share of whatever time is
left; a copy count that would start past the deadline is recorded as not attempted rather than
started and killed.

The result file is written after every cell, so a job killed mid-way still leaves what it had.
"""
import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

CODE = Path(__file__).resolve().parent
COPY_COUNTS = (512, 1024, 2048, 4096)

# a card that ran out of memory at one copy count will run out at every larger one, so the job
# stops climbing; these are the phrases the runtime uses when it does
OUT_OF_MEMORY_MARKERS = ("RESOURCE_EXHAUSTED", "Out of memory", "out of memory",
                         "OUT_OF_MEMORY", "CUDA_ERROR_OUT_OF_MEMORY")


def classify_failure(stderr_text):
    """Why a cell process died: the card's memory, or something else worth reading the log for."""
    return ("out_of_memory" if any(m in stderr_text for m in OUT_OF_MEMORY_MARKERS)
            else "failed")


def run_cell(python, n_copies, style, budget_seconds, cell_path, log_path):
    """Run one copy count as its own process and return its record."""
    # the cell writes its own result file; this process only reads it back or explains the death
    cmd = [python, str(CODE / "bench_cell.py"), "--n-copies", str(n_copies),
           "--style", style, "--budget-seconds", f"{budget_seconds:.0f}",
           "--out", str(cell_path)]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          timeout=budget_seconds + 900)
    log_path.write_text(f"$ {' '.join(cmd)}\n\n--- stdout ---\n{proc.stdout}\n"
                        f"--- stderr ---\n{proc.stderr}\n")
    if proc.returncode == 0 and cell_path.exists():
        return json.loads(cell_path.read_text())
    return {"status": classify_failure(proc.stderr), "n_copies": n_copies,
            "update_style": style, "returncode": proc.returncode,
            "wall_seconds": time.perf_counter() - t0,
            "stderr_tail": proc.stderr[-2000:], "log": log_path.name}


def main():
    """Measure every copy count on this node, under one wall-clock deadline."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--node-class", required=True)
    ap.add_argument("--cpus", type=int, required=True)
    ap.add_argument("--style", default="full_batch")
    ap.add_argument("--deadline-minutes", type=float, default=27.0)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--log-dir", type=Path, required=True)
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    stem = f"{args.node_class}__cpus-{args.cpus}__{args.style}"
    deadline = time.perf_counter() + args.deadline_minutes * 60
    record = {
        "node_class": args.node_class, "hostname": socket.gethostname(),
        "requested_cpus": args.cpus, "visible_cpus": len(os.sched_getaffinity(0)),
        "update_style": args.style, "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "cells": [],
    }
    out_path = args.out_dir / f"{stem}.json"

    # each remaining copy count gets an equal share of the time left, so an unexpectedly slow
    # small count cannot eat the budget of the large ones
    for i, n_copies in enumerate(COPY_COUNTS):
        remaining = deadline - time.perf_counter()
        if remaining <= 60:
            record["cells"].append({"status": "not_attempted_out_of_time",
                                    "n_copies": n_copies, "update_style": args.style})
            continue
        share = remaining / (len(COPY_COUNTS) - i)
        cell = run_cell(args.python, n_copies, args.style, share,
                        args.out_dir / f"{stem}__copies-{n_copies}.cell.json",
                        args.log_dir / f"{stem}__copies-{n_copies}.log")
        record["cells"].append(cell)
        record["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        out_path.write_text(json.dumps(record, indent=1))
        print(f"[{n_copies:>5} copies] {cell['status']}", flush=True)
        if cell["status"] == "out_of_memory":
            # every larger copy count needs strictly more memory on the same card
            for later in COPY_COUNTS[i + 1:]:
                record["cells"].append({"status": "not_attempted_smaller_count_ran_out_of_memory",
                                        "n_copies": later, "update_style": args.style})
            break

    record["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    out_path.write_text(json.dumps(record, indent=1))
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
