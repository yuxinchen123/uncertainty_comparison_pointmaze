#!/usr/bin/env python
"""Measure a condition the way the sweep actually runs it: N workers packed on one node at once.

Every number so far came from a single worker on an otherwise idle machine. Production runs ~46
single-threaded workers on a 24-core cortado node (two per physical core, `--ntasks-per-core=2`),
where the cores, the L3 and the memory bus are all shared. That is exactly the regime where an
allocator change can behave differently from the idle case — better (less kernel work per process)
or worse (more resident memory per node) — so the applied stack is worth what it is worth HERE, not
on the idle-node bench.

Each worker gets its own run: worker i takes marker i%3 and seed (marker seed + i), so the packed
job is a slice of the real sweep rather than the same run 46 times. Because the seeds are fixed by
worker index, worker i's record under `baseline` and under `combined` must still come out
identical — the validity check survives into the packed test.

Reported per condition: aggregate steps/s over the whole node (the number the sweep is paid in),
the per-worker median, and the identity verdict against the matching baseline worker.

Usage:
  python packed_bench.py --conditions baseline,combined --workers 46 --steps 3000 --out <json>
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
EFF = os.path.dirname(os.path.dirname(HERE))
PROJ = "/p/rlprojects/RND/07_reconstruction"
PY = "/p/rlprojects/RND/.venvs/exploration/bin/python"
sys.path.insert(0, HERE)
from run_conditions import CONDITIONS, marker_args, record_of, science_of, first_difference  # noqa: E402


def launch(idx, name, marker, marker_name, steps, eval_freq, data_dir, log_dir):
    """Start one packed worker; return (Popen, row-stub). Worker idx owns seed base+idx."""
    flags, env_extra = CONDITIONS[name]
    m = dict(marker)
    m["a_seed"] = int(marker["a_seed"]) + idx          # a distinct run per worker, as in the sweep
    out_dir = os.path.join(data_dir, f"{name}__w{idx:02d}")
    os.makedirs(out_dir, exist_ok=True)
    env = dict(os.environ)
    env.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                "PYTHONNOUSERSITE": "1", "TMPDIR": "/tmp"})
    env.update(env_extra)
    argv = [PY, os.path.join(HERE, "patched_entry.py")] + \
        marker_args(m, out_dir, steps, eval_freq) + flags
    log = open(os.path.join(log_dir, f"{name}__w{idx:02d}.log"), "w")
    proc = subprocess.Popen(argv, cwd=PROJ, env=env, stdout=log, stderr=subprocess.STDOUT)
    return proc, {"worker": idx, "config": marker_name, "out_dir": out_dir}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--markers", default=os.path.join(EFF, "data", "markers"))
    p.add_argument("--conditions", default="baseline,combined")
    p.add_argument("--workers", type=int, default=46)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--eval_freq", type=int, default=1500)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    markers = []
    for fn in sorted(os.listdir(args.markers)):
        if fn.endswith(".json"):
            with open(os.path.join(args.markers, fn)) as fh:
                markers.append((fn[:-5], json.load(fh)))

    job = os.environ.get("SLURM_JOB_ID", "local")
    data_dir = os.path.join(EFF, "data", f"packed_{job}")
    log_dir = os.path.join(EFF, "logs", f"packed_{job}")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    results, baseline_science = [], {}
    for name in [c.strip() for c in args.conditions.split(",") if c.strip()]:
        procs = []
        t0 = time.time()
        for i in range(args.workers):
            marker_name, marker = markers[i % len(markers)]
            procs.append(launch(i, name, marker, marker_name, args.steps, args.eval_freq,
                                data_dir, log_dir))
        rows = []
        for proc, stub in procs:
            rc = proc.wait()
            rec = record_of(stub["out_dir"])
            rt = rec.get("runtime_seconds") if rec else None
            stub.update({"rc": rc, "runtime_s": None if rt is None else round(rt, 1),
                         "steps_per_s": None if not rt else round(args.steps / rt, 3)})
            key = (name, stub["worker"])
            if name == "baseline":
                baseline_science[stub["worker"]] = science_of(rec)
            base = baseline_science.get(stub["worker"])
            stub["validity"] = ("no baseline" if base is None else
                                "IDENTICAL" if first_difference(base, science_of(rec)) is None else
                                f"DIFFERS at {first_difference(base, science_of(rec))}")
            rows.append(stub)
        wall = time.time() - t0
        ok = [r["steps_per_s"] for r in rows if r["steps_per_s"]]
        ok.sort()
        agg = args.workers * args.steps / wall
        ident = sum(1 for r in rows if r["validity"] == "IDENTICAL")
        print(f"[packed] {name:<10} workers={args.workers} wall={wall:7.1f}s "
              f"aggregate={agg:7.2f} steps/s  per-worker median={ok[len(ok)//2] if ok else None} "
              f"identical={ident}/{len(rows)}", flush=True)
        results.append({"condition": name, "workers": args.workers, "steps": args.steps,
                        "wall_s": round(wall, 1), "aggregate_steps_per_s": round(agg, 3),
                        "per_worker_median": ok[len(ok) // 2] if ok else None,
                        "identical": ident, "rows": rows})
        with open(args.out, "w") as fh:
            json.dump({"node": os.environ.get("SLURMD_NODENAME", ""), "job": job,
                       "results": results}, fh, indent=1)

    if len(results) > 1:
        base = results[0]["aggregate_steps_per_s"]
        for r in results[1:]:
            print(f"[packed] {r['condition']} vs {results[0]['condition']}: "
                  f"x{r['aggregate_steps_per_s'] / base:.4f} aggregate", flush=True)
    print(f"[packed] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
