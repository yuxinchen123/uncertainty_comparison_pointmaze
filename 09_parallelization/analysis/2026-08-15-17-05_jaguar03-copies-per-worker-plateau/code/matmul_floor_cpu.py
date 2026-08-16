"""The matrix work of one training iteration, timed on its own, on ordinary processor cores.

This is the processor analogue of `analysis/ceiling/code/matmul_floor.py`, which is what the
graphics-processor side of this report measures itself against, and it follows that script's
method exactly: every matrix multiply of one iteration is timed on its own, forward passes once
and backward passes three times (the layer's output, the gradient with respect to its input, and
the gradient with respect to its weights), and the sum is what the iteration would cost if the
matrix multiplies were the only work and nothing else took any time at all.

Two differences from the graphics-processor script, both forced by the machine:

  same load     a processor's matrix multiply is only as fast as the memory system lets it be, and
                the memory system is shared. So the floor is measured with the SAME number of
                worker processes as the training run it will be compared against, each with one
                thread and its own copies, all held at a barrier so they measure at once. A floor
                measured on an idle node would be a floor for a machine nobody is using.
  same shapes   the copies per worker replace the graphics processor's fixed 128 copies, since
                that is the knob this sweep moves.

What the resulting percentage means: it is the share of the training iteration that the matrix
work's own best case accounts for. A high percentage says the iteration is mostly matrix
arithmetic and there is little left to win; a low one says most of the time is elsewhere.

Usage:
  python matmul_floor_cpu.py --copies 1 16 64 --procs-list 112 224 --styles full_batch
"""
import argparse
import json
import multiprocessing as mp
import platform
import sys
import time
from pathlib import Path

RUN = Path(__file__).resolve().parent.parent
BASE = RUN.parent.parent
sys.path.insert(0, str(RUN / "code"))

# layer widths, from ppo/research/ppo_rnd_algorithm_spec.md section 3, the same lists the
# graphics-processor floor script uses
ACTOR = [(4, 64), (64, 64), (64, 2)]
CRITIC = [(4, 64), (64, 64), (64, 2)]      # two 1-wide heads, packed into one width-2 multiply
TARGET = [(4, 256), (256, 128)]
PREDICTOR = [(4, 256), (256, 128), (128, 128)]
T, N_ENVS, B = 128, 4, 512                 # rollout steps, rows per step per copy, batch rows
MINIBATCHES = 16                           # update steps when the convention is sixteen updates
BARRIER = None


def note(line):
    """Append one line to the run's progress file and to standard output, flushed immediately."""
    from bench_copies_per_worker import note as bench_note
    bench_note(line)


def bmm_time(cache, c, m, k, n, target_seconds=0.15):
    """Median seconds for one batched multiply of (c, m, k) by (c, k, n), on one thread.

    The repeat count is chosen from a first timing so that every shape is measured over about the
    same amount of work: the small rollout multiplies need thousands of repeats to be measurable
    and the wide update multiplies need a handful.
    """
    import torch
    key = (c, m, k, n)
    if key in cache:
        return cache[key]
    a = torch.randn(c, m, k)
    b = torch.randn(c, k, n)
    out = torch.empty(c, m, n)
    torch.bmm(a, b, out=out)
    t0 = time.perf_counter()
    torch.bmm(a, b, out=out)
    once = max(time.perf_counter() - t0, 1e-7)
    repeats = max(3, min(2000, int(target_seconds / once)))
    samples = []
    for _ in range(5):
        t0 = time.perf_counter()
        for _ in range(repeats):
            torch.bmm(a, b, out=out)
        samples.append((time.perf_counter() - t0) / repeats)
    samples.sort()
    cache[key] = samples[len(samples) // 2]
    return cache[key]


def layer_cost(cache, c, m, k, n, backward):
    """One layer's isolated time: forward, plus the two backward multiplies when it is trained.

    before: a 64-wide layer on 128 rows, trained
    after:  forward (128,64)x(64,64) + input gradient (128,64)x(64,64) + weight gradient
            (64,128)x(128,64), summed
    """
    fwd = bmm_time(cache, c, m, k, n)
    if not backward:
        return fwd
    return fwd + bmm_time(cache, c, m, n, k) + bmm_time(cache, c, k, m, n)


def iteration_floor(copies, style):
    """The summed isolated time of every matrix multiply in one training iteration."""
    cache = {}
    # the rollout: T sequential steps, each running the actor forward on N rows per copy
    total = T * sum(layer_cost(cache, copies, N_ENVS, k, n, False) for k, n in ACTOR)
    # the wide passes after the rollout: the critic on both observations, the intrinsic networks
    for rows, group in ((2 * B, CRITIC), (B, TARGET), (B, PREDICTOR), (B, TARGET)):
        total += sum(layer_cost(cache, copies, rows, k, n, False) for k, n in group)
    # the update: one pass over the whole batch, or sixteen passes over a sixteenth of it
    rows, passes = ((B, 1) if style == "full_batch" else (B // MINIBATCHES, MINIBATCHES))
    for group in (ACTOR, CRITIC, PREDICTOR):
        total += passes * sum(layer_cost(cache, copies, rows, k, n, True) for k, n in group)
    return total


def pool_init(barrier):
    """Give every worker the barrier that starts the measurement for all of them at once."""
    global BARRIER
    BARRIER = barrier


def floor_worker(args):
    """One worker: warm the caches, wait for every other worker, then time its own matrix work."""
    copies, style = args
    import torch
    torch.set_num_threads(1)
    iteration_floor(copies, style)          # warm-up, so no allocation is inside the timed sum
    if BARRIER is not None:
        BARRIER.wait(timeout=3600)
    return iteration_floor(copies, style)


def run_floor(procs, copies, style):
    """The matrix-work floor for one setting, measured with the whole machine doing the same."""
    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(procs)
    t0 = time.perf_counter()
    with ctx.Pool(procs, initializer=pool_init, initargs=(barrier,)) as pool:
        seconds = pool.map(floor_worker, [(copies, style)] * procs)
    seconds.sort()
    row = {"workers": procs, "n_copies": copies, "style": style,
           "floor_seconds_median_worker": seconds[len(seconds) // 2],
           "floor_seconds_fastest_worker": seconds[0],
           "floor_seconds_slowest_worker": seconds[-1],
           "wall_seconds": time.perf_counter() - t0}
    note(f"[floor] procs={procs} copies={copies} style={style} "
         f"floor={row['floor_seconds_median_worker']:.4f}s per iteration "
         f"wall={row['wall_seconds']:.0f}s")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--copies", type=int, nargs="+", required=True)
    ap.add_argument("--procs-list", type=int, nargs="+", default=[112, 224])
    ap.add_argument("--styles", nargs="+", default=["full_batch", "epoch_minibatch"])
    args = ap.parse_args()
    note(f"matrix-work floor on {platform.node()}")

    out = RUN / "data" / "matmul_floor_cpu.json"
    rows = json.loads(out.read_text()) if out.exists() else []
    # copies outermost, so the cheap settings of every series are measured before the dear ones
    for copies in args.copies:
        for procs in args.procs_list:
            for style in args.styles:
                rows.append(run_floor(procs, copies, style))
                out.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
