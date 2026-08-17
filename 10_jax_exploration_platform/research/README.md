# research

Measurement rounds and their write-ups. Never production code — accepted code moves to `src/`.

| folder | what lives there |
|---|---|
| `systems/` | rounds about the platform itself: the environment, the agent, the whole iteration |
| `algorithms/<family>/` | one bonus family's fusion work: `ledger.md` plus `rounds/<date>-rNN-<slug>/` |
| `proposals/` | algorithms not yet built, in the same round shape, promoted through the same gate |

A family's rounds end when it passes the gate the platform sets for every new bonus: **total
environment steps per second at least PPO with random network distillation's, at the same copy
count, on the same card, with the bar re-measured in the same session rather than quoted from a
file** — plus the correctness gates that family already had to hold.

Times below are Pacific with a `PT` marker; the machines run on Eastern and the times were
converted for display.

## Round summaries

### The first batch's bonuses against the distillation bar — 2026-08-16 19:05–19:44 PT

Fork `algo/visit-count`, H100 NVL on serval05 under the lock, jax 0.11.0, platform commit
`f3fd968`. Raw output `benchmark_runs/2026-08-16_visit-count-gate/`; ledgers
[`algorithms/visit_count/ledger.md`](algorithms/visit_count/ledger.md) and
[`algorithms/none/ledger.md`](algorithms/none/ledger.md).

All three arms of the first batch pass the gate **with no code change at all**, each at the copy
count its own science run uses, each against a bar re-measured beside it in the same process:

| bonus | copies | seconds per iteration | total env steps per second | env steps per second per copy | hours per million steps per copy | peak device memory | against the bar | rounds won |
|---|---|---|---|---|---|---|---|---|
| `rnd_next_state` (the bar) | 8,448 | 0.08402 | 51.48 M | 6.09 k | 0.0456 | 28.16 GiB | 1.000 | — |
| `gt_position_velocity_sqrt` | 8,448 | 0.03450 | 125.38 M | 14.84 k | 0.0187 | 7.78 GiB | **2.436** | 11 of 11 |
| `gt_position_velocity_linear` | 8,448 | 0.03427 | 126.21 M | 14.94 k | 0.0186 | 6.89 GiB | **2.449** | 11 of 11 |
| `rnd_next_state` (the bar) | 768 | 0.01116 | 35.24 M | 45.89 k | 0.0061 | 2.98 GiB | 1.000 | — |
| `none` | 768 | 0.00699 | 56.27 M | 73.27 k | 0.0038 | 0.76 GiB | **1.597** | 11 of 11 |

The round's own finding is why no change was needed. A phase profile written for the `full_batch`
update style put the **visit-count bonus's entire share of an iteration at about 1.2 percent** —
its index arithmetic, scatter, gather and power timed on their own come to 0.412 ms of a 33.8 ms
iteration, the same figure in two separate runs — so 1.2 percent was the ceiling on every fusion
idea that had been queued, and all of them were discarded against it rather than measured. The
prediction going in had been the opposite: that the scatter and gather would put the visit-count
arms behind. What makes them cheap is locality — over one rollout a copy reaches about 20 of its
10,800 table entries — while random network distillation pays two networks over a 4.3-million-row
batch, which is 50 ms and 20.4 GiB of its 28.2 GiB.

Two consequences beyond the gate:

1. **The batch is cheaper than planned.** At these rates 10 M steps per copy is about 11 minutes
   per visit-count arm, 27 minutes for the distillation arm and 3 minutes for the no-bonus arm, on
   one H100 — which is the proportionality clause's case for not spreading the batch thinly across
   many cards.
2. **Further throughput work belongs to `systems/`, not to a bonus family.** With no bonus at all
   the iteration is still 33.4 ms at 8,448 copies: 40 percent update, 60 percent rollout and
   post-processing.

A caveat recorded for whoever profiles next: `benchmarks/harness/profile_jax_phases.py` subtracts a
fixed sixteen-step gradient probe from the real update, which is one step under `full_batch`, so
its clip and optimizer rows come out negative in that style. Use
`benchmarks/bonus/profile_visit_count_phases.py`, which times each piece instead of subtracting
unlike things.
