# Round 1 — the fused form against the distillation bar, at the size the science run will use

Written 2026-08-16 18:54 PT, before any measurement at the real size.

## The question

The first batch's visit-count arms run 3 learning rates x 11 intrinsic weights x 256 copies =
**8,448 copies**, one update per batch, 128 rollout steps of 4 environments each. The gate they
have to pass is fixed: **total environment steps per second at least random network
distillation's, at the same copy count, on the same card, re-measured in the same session.**

## What was expected, and why

The expectation going into this round was that the visit-count arms would come out **below** the
bar and that the round would be about closing the gap. The reasoning behind that expectation:

1. **The scatter and the gather are memory traffic distillation does not have.** The table is one
   `int32` array of 10,800 entries per copy — 365 MB across 8,448 copies, far past any cache, so
   every counted row is a read-modify-write to a scattered address in main memory, and the gather
   that follows reads those same addresses again.
2. **Distillation's extra is two small networks**, and the systems study behind
   `09_parallelization` found that at these copy counts the iteration is bound by the bytes it
   moves rather than the programs it issues. Small dense matrix multiplications move their inputs
   once and are otherwise arithmetic, which is the cheap side of that trade.

So the predicted ordering was: distillation faster, visit count behind by whatever the scattered
traffic costs.

## What would have followed if that had held

The directions listed for the fusion work, in the order they would have been tried: the index
dtype and the table's layout; folding the scatter into the pass that already walks the rollout's
observations; reusing the scatter's indices for the gather; the unroll factor of the update scan;
and splitting the table's last axis so the velocity bins of one cell sit together.

## How it is measured

`benchmarks/bonus/bench_bonus_against_rnd.py`, at the shipped sweep shape: both arms alternate
round by round inside one process on one card, the order flips every round, every iteration is
waited for, the medians are taken over 11 rounds of 10 iterations, and the round-by-round sign
test is reported beside the medians. Peak device memory comes from separate one-arm processes,
because the allocator's high-water mark belongs to the process and cannot be split between two
arms sharing one. After the rounds the arm's own state is read back: the count table has to hold
exactly as many counts as rows were scattered, so a scatter the compiler had deleted would be
caught rather than measured.
