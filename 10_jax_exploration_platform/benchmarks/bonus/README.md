# benchmarks/bonus

What each bonus family costs, measured against the family beside it rather than against a number
from another day.

| file | what it measures |
|---|---|
| `bench_none_vs_rnd.py` | seconds per iteration of PPO with no bonus against PPO with random network distillation, paired round by round in one process on one card |

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
it reports no memory. The per-family throughput gates belong to the fusion work that follows.
