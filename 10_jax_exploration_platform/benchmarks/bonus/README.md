# benchmarks/bonus

What each bonus family costs, measured against the family beside it rather than against a number
from another day.

| file | what it measures |
|---|---|
| `bench_bonus_against_rnd.py` | **the gate**: any bonus against random network distillation at the copy count its science run uses, in the shipped sweep shape, paired round by round with the order flipped every round; `--solo` runs one arm alone for its peak device memory, which cannot be split between two arms sharing a process |
| `profile_visit_count_phases.py` | where a visit-count iteration's time goes under `full_batch`: the whole iteration with and without the bonus, the update stage alone, and the bonus's index arithmetic, scatter and gather timed separately on indices a real rollout produced |
| `bench_none_vs_rnd.py` | the first, quick version of the same paired comparison, for no bonus against distillation at one copy count; superseded by `bench_bonus_against_rnd.py`, which adds the sweep shape, the memory column and the check that the bonus under test actually ran |

`benchmarks/harness/profile_jax_phases.py` is for `epoch_minibatch` only: it subtracts a fixed
sixteen-step gradient probe from the real update, which is one step under `full_batch`, so its clip
and optimizer rows come out negative in that style. Its whole-iteration, update-stage and
environment-alone rows are right in both.

Result files land in `results/` and are not committed (the repository ignores `*.json`); the
tables below carry what they said.

Times are Pacific with a `PT` marker; the machines run on Eastern and the times were converted for
display.

## No bonus against random network distillation, 2026-08-16 17:41 PT

H100 NVL on serval05, under the lock; jax 0.11.0, `/p/rlprojects/RND/.venvs/platform_jax`.
128 copies x 4 environments x 128 rollout steps, one update per batch, 11 paired rounds of 10
iterations each, timed sync (every iteration waited for).

| bonus | copies | seconds per iteration | total env steps per second | env steps per second per copy | hours per million steps per copy |
|---|---|---|---|---|---|
| `rnd_next_state` | 128 | 0.005917 | 11.08 M | 86.5 k | 0.00321 |
| `none` | 128 | 0.004979 | 13.16 M | 102.8 k | 0.00270 |

No bonus reaches 1.188 times the throughput and won **11 of 11** paired rounds — every round, not
a median that happened to fall the right way. That is the other half of the select-before-compiling
claim: `tests/bonuses/test_none_has_no_bonus_arithmetic.py` shows the bonus's arithmetic is absent
from the compiled program, and this shows the absence is worth time on the card.

This is a quick paired check, not the rigorous measurement: it is one copy count on one card, and
it reports no memory. The per-family throughput gates are the section below.

## The first batch's gate, 2026-08-16 19:05 to 19:44 PT

H100 NVL on serval05, under the lock; jax 0.11.0, `/p/rlprojects/RND/.venvs/platform_jax`; platform
commit `f3fd968`. Each arm at the copy count its own science run uses, in the shipped sweep shape
(learning rates x intrinsic weights x 256 copies per cell), one update per batch, 128 rollout steps
of 4 environments per copy. Medians over 11 paired rounds of 10 iterations, timed sync; peak memory
from separate one-arm processes. Raw output: `benchmark_runs/2026-08-16_visit-count-gate/`.

| bonus | copies | seconds per iteration | total env steps per second | env steps per second per copy | hours per million steps per copy | peak device memory | against the bar | rounds won |
|---|---|---|---|---|---|---|---|---|
| `rnd_next_state` (the bar) | 8,448 | 0.08402 | 51.48 M | 6.09 k | 0.0456 | 28.16 GiB | 1.000 | — |
| `gt_position_velocity_sqrt` | 8,448 | 0.03450 | 125.38 M | 14.84 k | 0.0187 | 7.78 GiB | **2.436** | 11 of 11 |
| `gt_position_velocity_linear` | 8,448 | 0.03427 | 126.21 M | 14.94 k | 0.0186 | 6.89 GiB | **2.449** | 11 of 11 |
| `rnd_next_state` (the bar) | 768 | 0.01116 | 35.24 M | 45.89 k | 0.0061 | 2.98 GiB | 1.000 | — |
| `none` | 768 | 0.00699 | 56.27 M | 73.27 k | 0.0038 | 0.76 GiB | **1.597** | 11 of 11 |

The bar was re-measured in each paired session rather than quoted; its two rows at 8,448 copies
agree to 0.1 percent. Every arm passes with no code change. Why, and what was discarded because of
it: `research/algorithms/visit_count/rounds/2026-08-16-r01-fused-form-against-the-distillation-bar/verdict.md`.
