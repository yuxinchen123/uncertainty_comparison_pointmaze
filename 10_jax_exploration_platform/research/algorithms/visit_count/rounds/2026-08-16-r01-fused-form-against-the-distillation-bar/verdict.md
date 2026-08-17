# Verdict — round 1: the fused form passes the gate as built, by 2.44 times

Measured 2026-08-16, 19:05 to 20:21 PT, H100 NVL on serval05 under the lock; jax 0.11.0 from
`/p/rlprojects/RND/.venvs/platform_jax`; platform commit `f3fd968`. Times are Pacific with a `PT`
marker (the machines run on Eastern and the times were converted for display). Raw output:
`benchmark_runs/2026-08-16_visit-count-gate/`; the numbers below are also in `result.json` beside
this file.

**Nothing was changed.** The prediction the round was built on — that the scatter and the gather
would put the visit-count arms behind random network distillation — was wrong, and wrong by a
wide margin in the other direction.

## The gate

Total environment steps per second at the copy count the science run uses, one update per batch,
128 rollout steps of 4 environments per copy, the shipped sweep shape. Both arms of each
comparison alternate round by round in one process on one card, the order flips every round, every
iteration is waited for, and the medians are over 11 rounds of 10 iterations. Peak memory is from
separate one-arm processes.

| bonus | copies | seconds per iteration | total env steps per second | env steps per second per copy | hours per million steps per copy | peak device memory | against the bar | rounds won |
|---|---|---|---|---|---|---|---|---|
| `rnd_next_state` (the bar) | 8,448 | 0.08402 | 51.48 M | 6.09 k | 0.0456 | 28.16 GiB | 1.000 | — |
| `gt_position_velocity_sqrt` | 8,448 | 0.03450 | 125.38 M | 14.84 k | 0.0187 | 7.78 GiB | **2.436** | 11 of 11 |
| `rnd_next_state` (the bar) | 8,448 | 0.08394 | 51.53 M | 6.10 k | 0.0455 | 28.16 GiB | 1.000 | — |
| `gt_position_velocity_linear` | 8,448 | 0.03427 | 126.21 M | 14.94 k | 0.0186 | 6.89 GiB | **2.449** | 11 of 11 |

Both presets pass. The two bar rows are two separate re-measurements of random network
distillation, one in each paired session, and they agree to within 0.1 percent (84.02 against
83.94 ms), which is what a bar re-measured rather than quoted is for.

The margin needs no sign test to be believable — it is a factor of two and a half, not a couple of
percent — but the sign test was run anyway and every one of the 22 paired rounds went the same
way.

## Why the prediction was wrong

The phase profile at 8,448 copies, `full_batch`, written for this update style rather than
inherited from the sixteen-step one (`benchmarks/bonus/profile_visit_count_phases.py`). It was run
twice, and both runs are given, because one of its rows is a difference of two large numbers and
its noise is only visible across repeats:

| run 1, ms | run 2, ms | share of the iteration | phase |
|---|---|---|---|
| 33.858 | 33.802 | 100% | whole iteration, visit count |
| 33.441 | 33.566 | 99% | whole iteration, **no bonus at all** |
| 0.417 | 0.236 | **0.7 to 1.2%** | the bonus's whole share (the difference of the two above) |
| 13.614 | 13.536 | 40% | update stage alone (one gradient step and the optimizer) |
| 20.244 | 20.266 | 60% | rollout and post-processing (whole minus update) |
| 0.178 | 0.179 | 0.5% | index arithmetic alone |
| 0.380 | 0.383 | 1.1% | index and scatter |
| 0.412 | 0.413 | 1.2% | index, scatter, gather and power |

Two things follow, and they are the whole result.

1. **The visit-count bonus costs about 1.2 percent of an iteration, and no more.** The two ways of
   measuring it are not equally trustworthy and should not be presented as if they were. Timing
   the bonus's three operations on their own, on indices a real rollout produced, gives **0.412
   and 0.413 ms** — the same number twice, to a microsecond. Differencing the whole program with
   the bonus against the whole program without it gives **0.417 and 0.236 ms** — a spread of
   0.18 ms, which is what subtracting two 33.8 ms measurements of two separately compiled programs
   is worth. The standalone figure is the one to quote, and the difference brackets it rather than
   confirming it. Both say the same thing at the resolution that matters: the bonus is around one
   percent of the iteration, and it is certainly not ten.
2. **So roughly 1.2 percent is the ceiling on every optimisation that was going to be tried.**
   Removing the entire bonus — index arithmetic, table, scatter, gather and power — would buy
   about 1.2 percent. Index dtypes, table layout, folding the scatter into the observation pass,
   reusing indices between the scatter and the gather, and splitting the table's last axis all
   live inside that 1.2 percent, and most of them inside the 0.2 ms the scatter itself takes.

The reason the traffic is cheap is locality, and it can be stated as a number: over one rollout a
copy reaches about 20 distinct table entries out of its 10,800, because the ball moves
continuously and 512 consecutive steps stay in a handful of cells. Across 8,448 copies that is a
few tens of thousands of distinct entries, a working set of tens of megabytes of cache lines
rather than the 365 MB the table occupies. (The 20 is measured on a rollout from the start state;
after 113 training iterations a copy's table held 239 non-zero entries in total, so the per-rollout
figure stays in the low hundreds at worst — and the paired rounds' times were flat to within 3
percent while those tables filled, so nothing drifted as the copies spread out.)

What random network distillation pays instead is two networks over the whole 4.3-million-row
batch, forward for the target, forward and backward for the predictor. That is 50 ms — more than
the entire rest of the iteration — and 20.4 GiB of the 28.2 GiB the arm holds.

## Proof the thing under test ran

The compiler is free to delete arithmetic nothing consumes, and this project has twice measured a
variant that had been optimised away, so the benchmark reads the arm's own state back afterwards:

- The count table held **57,856 counts per copy, minimum and maximum alike**, after 113 iterations
  of 512 rows — exactly one count per counted row, in every one of the 8,448 copies. A scatter
  that had been deleted, run once, or run on the wrong rows could not produce that.
- The intrinsic reward it produced is inside the bonus's definition, `(0, 1]`, and separates the
  two presets the way their exponents say it must: 0.0261 mean for `1/sqrt(n)` against 0.00234 for
  `1/n` on the same trajectories.
- The no-bonus arm's intrinsic reward is exactly 0.0, so its own gate row is not measuring a
  bonus that quietly survived.

## What was discarded, and why

Every direction the round was set up to test, discarded unmeasured, on the strength of the ceiling
above rather than on a measurement of its own:

1. **Index dtype and table layout** — the index array is `[C, 512]`, 17 MB at `int32`; halving it
   cannot pay inside a 0.178 ms phase.
2. **Folding the scatter into the pass that already walks the rollout's observations** — the pass
   it would fold into is part of the 20.2 ms rollout-and-post-processing block, and the whole
   thing being folded in is 0.2 ms.
3. **Reusing the scatter's indices for the gather** — they are already the same array,
   `idx`, computed once and used by both; there was nothing to reuse that is not reused.
4. **Splitting the table's last axis to `[C, rows*cols, 100]`** — a reshape of the same memory in
   the same order; the addresses the scatter touches would be identical.
5. **The unroll factor of the update scan** — belongs to the agent, not to this family, and one
   update per batch is one step, so there is no scan to unroll in this update style.

A sixth was considered and is worth writing down because it is the one that would matter if the
gate had been close: the scatter is a two-dimensional `at[copy_rows, idx].add(...)`, and flattening
it to a one-dimensional scatter into `counts.reshape(-1)` with a per-copy offset added into the
index is sometimes a better lowering. It was not measured, for the same reason as the rest — the
whole phase it lives in is 0.2 ms of a 33.9 ms iteration.

## What this round leaves for later

The gate is passed and the family ships as it is. The one thing the profile says is worth
attention is not about this family at all: at 8,448 copies the iteration is 40 percent update and
60 percent rollout and post-processing, with no bonus in either, so any further work on the
platform's throughput belongs to `research/systems/`, not here.
