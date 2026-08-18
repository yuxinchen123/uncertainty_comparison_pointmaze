"""Fixed experiment runner (program.md): runs the current code/method.py over (point set x
seed) cells under the fixed protocol, writes per-cell JSON records with atomic checkpointed
flushes, computes the metric battery, and prints the '---' summary block.

One experiment:
    /p/rlprojects/RND/.venvs/exploration/bin/python run_experiment.py \
        --out ../experiments/<stamp>_<name> [--envs cell_midpoints,center_square] \
        [--seeds 10] [--n_steps 4096] [--regime uniform_fullbatch] [--procs 16]

Re-running with the same --out resumes: cells whose record is complete are skipped (the
project's resumable-by-construction rule). The runner never reads method internals — only the
Method interface documented in method.py.
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decay_harness.metrics import compute_metrics, summary_block
from decay_harness.points import (POINT_SET_CHOICES, checkpoint_steps, nonuniform_probabilities,
                                  point_set)

CELL_TIME_CAP = float(os.environ.get("CELL_TIME_CAP", 900.0))  # seconds per (env, seed) cell;
# past it the cell aborts as overtime. Overridable by environment variable for GPU/image
# domains whose legitimate cell times differ (an infrastructure knob, not a protocol change).
NONUNIFORM_BATCH = 32  # batch size of the nonuniform regime's sampled updates


def cell_path(out_dir: str, env: str, seed: int) -> str:
    """Record path for one (point set, seed) cell."""
    return os.path.join(out_dir, f"{env}_seed{seed:03d}.json")


def run_cell(args) -> str:
    """Worker: run one (env, seed) cell of the current method and write its record; returns a
    one-line status string for the launcher log."""
    out_dir, env, seed, n_steps, regime = args
    # single-thread torch per worker so --procs workers pack cleanly onto the allocation
    torch.set_num_threads(1)
    path = cell_path(out_dir, env, seed)
    # resume: a complete record is final — skip (idempotent re-runs, append-only convention)
    if os.path.exists(path):
        with open(path) as fh:
            if json.load(fh).get("completed"):
                return f"skip {env} seed {seed} (complete)"

    # import the experiment's SNAPSHOT of method.py (submit_experiment.sh copies it at submit
    # time), so editing code/method.py after submission can never change a queued experiment
    sys.path.insert(0, out_dir)
    from method import Method  # imported in the worker so a crash poisons only this cell
    t0 = time.time()
    pts = point_set(env)
    x_all = torch.as_tensor(pts)
    # vector domains pass their width; image domains pass the full trailing shape tuple
    # before: pts (108, 4) -> obs_dim 4;  pts (512, 84, 84) -> obs_dim (84, 84)
    obs_dim = pts.shape[1] if pts.ndim == 2 else tuple(pts.shape[1:])
    m = Method(seed, obs_dim=obs_dim)
    cps = checkpoint_steps(n_steps)

    # nonuniform regime: fixed sampling law + its own keyed numpy stream, so the visit sequence
    # is reproducible per (env, seed) and independent of the method's own randomness
    probs = nonuniform_probabilities(pts) if regime == "nonuniform" else None
    # keyed stream (reproducible-seeding rule): builtin hash() is salted per process, so the
    # seed comes from sha256 of the stable cell identity instead
    import hashlib
    key = f"{env}::{seed}::visits".encode()
    rng = np.random.default_rng(int(hashlib.sha256(key).hexdigest(), 16) & 0x7FFFFFFF)
    counts = np.zeros(pts.shape[0], dtype=np.int64)

    record = {"completed": False, "diverged": False, "overtime": False,
              "method": Method.name, "point_set": env, "regime": regime, "a_seed": seed,
              "n_steps": n_steps, "n_points": int(pts.shape[0]),
              "checkpoint_steps": [0] + cps, "bonus": [], "visit_counts": [],
              "runtime_seconds": 0.0}

    def flush(completed: bool) -> None:
        """Atomic tmp + os.replace rewrite of the cell record (checkpointed-logging rule)."""
        record["completed"] = completed
        record["runtime_seconds"] = time.time() - t0
        with open(path + ".tmp", "w") as fh:
            json.dump(record, fh)
        os.replace(path + ".tmp", path)

    def snapshot() -> bool:
        """Record the readout on the full point set; False if non-finite (divergence)."""
        b = m.bonus(x_all)
        if not np.isfinite(b).all():
            record["diverged"] = True
            return False
        record["bonus"].append(b.tolist())
        record["visit_counts"].append(counts.tolist())
        return True

    snapshot()  # step 0: the initial bonus field (b_i(0); metric requirement 1 reads this row)
    flush(False)
    cp_set = set(cps)
    for t in range(1, n_steps + 1):
        # the regime picks the batch; the method only ever sees the batch tensor
        if regime == "uniform_fullbatch":
            batch_idx = np.arange(pts.shape[0])
        else:
            batch_idx = rng.choice(pts.shape[0], size=NONUNIFORM_BATCH, p=probs)
        counts[batch_idx] += 1
        m.update(x_all[batch_idx])
        if t in cp_set:
            if not snapshot():
                break
            flush(False)
            if time.time() - t0 > CELL_TIME_CAP:
                record["overtime"] = True
                break
    flush(True)
    tag = "DIVERGED" if record["diverged"] else ("OVERTIME" if record["overtime"] else "ok")
    return f"done {env} seed {seed}: {tag}, {record['runtime_seconds']:.1f}s"


def main() -> None:
    """Launch every (env, seed) cell in a process pool, then compute and print the metrics."""
    p = argparse.ArgumentParser(description="Run one 11_decay_rate experiment (fixed protocol)")
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--envs", type=str, default="cell_midpoints,center_square")
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--n_steps", type=int, default=4096)
    p.add_argument("--regime", type=str, default="uniform_fullbatch",
                   choices=["uniform_fullbatch", "nonuniform"])
    p.add_argument("--procs", type=int, default=16)
    args = p.parse_args()

    envs = [e for e in args.envs.split(",") if e]
    for e in envs:
        if e not in POINT_SET_CHOICES:
            raise ValueError(f"unknown point set {e!r}; choices: {POINT_SET_CHOICES}")
    os.makedirs(args.out, exist_ok=True)
    # local runs without a submit-time snapshot: freeze the current method.py now
    snap = os.path.join(args.out, "method.py")
    if not os.path.exists(snap):
        import shutil
        shutil.copy2(os.path.join(os.path.dirname(os.path.abspath(__file__)), "method.py"), snap)
    t0 = time.time()

    # fan the cells over the pool; each worker imports method.py independently
    cells = [(args.out, e, s, args.n_steps, args.regime) for e in envs for s in range(args.seeds)]
    with Pool(processes=args.procs) as pool:
        for line in pool.imap_unordered(run_cell, cells):
            print(f"[progress] {line}", flush=True)

    # load every record back and run the fixed metric battery over all cells together
    records = []
    for e in envs:
        for s in range(args.seeds):
            with open(cell_path(args.out, e, s)) as fh:
                records.append(json.load(fh))
    metrics = compute_metrics(records)
    metrics["experiment"] = {"envs": envs, "seeds": args.seeds, "n_steps": args.n_steps,
                             "regime": args.regime, "method": records[0]["method"],
                             "wall_seconds": time.time() - t0}
    with open(os.path.join(args.out, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=1)
    print(summary_block(metrics, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
